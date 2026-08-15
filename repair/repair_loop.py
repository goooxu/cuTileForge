"""Multi-turn compile-feedback repair loop.

Phase two showed ability tracks the training data almost linearly, and that
convolution has no seed at all: 263 sampled conv kernels, zero correct, even at
rank 3 with kernel_size=1. Rejection sampling cannot bootstrap from nothing, so
this feeds the compiler's own diagnostic back to the model and asks it to fix
its kernel, up to a few rounds.

Two reasons to expect this to work: 85% of "wrote real cuTile but wrong"
failures die at compile or import rather than on numerics, and cuTile's errors
name the file, line and column.

The loop advances in lockstep -- sample the whole batch, verify the whole batch,
build repair turns for the failures, repeat -- so the server stays saturated
instead of idling on one candidate at a time.

Usage:
    python3 repair/repair_loop.py --level 93 --samples 4 --max-rounds 3 \\
        --out /ws/runs/repair_l93
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify"))

from prompts import build_repair_message  # noqa: E402
from worker import VerifierPool  # noqa: E402


def extract_code(text: str):
    from kernelbench.utils import extract_best_code
    return extract_best_code(text, ["python", "cpp"])


class Chat:
    """Thin OpenAI-compatible client for the locally served model."""

    def __init__(self, base_url: str, model: str, temperature: float,
                 top_p: float, top_k: int, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI(api_key="local-no-auth", base_url=base_url,
                             timeout=1800, max_retries=2)
        self.model = model
        self.kw = dict(temperature=temperature, top_p=top_p,
                       max_tokens=max_tokens, extra_body={"top_k": top_k})

    def complete(self, messages):
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, **self.kw)
        return r.choices[0].message.content or ""


def sample_batch(chat: Chat, message_lists, concurrency: int):
    """Fire a batch of chat completions concurrently, preserving order."""
    out = [None] * len(message_lists)

    def one(i):
        try:
            return i, chat.complete(message_lists[i])
        except Exception as e:
            return i, "__ERROR__ %s: %s" % (type(e).__name__, str(e)[:200])

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, text in ex.map(one, range(len(message_lists))):
            out[i] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--samples", type=int, default=4, help="Candidates per task.")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="Repair rounds after the initial attempt.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--verify-workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--prompt-tier", default="cutile_docs",
                    help="Prompt composition for the first turn. Must match what "
                         "the model was trained on: running a model trained on "
                         "cutile_concepts with the full reference is out of "
                         "distribution and measures nothing useful. Default keeps "
                         "the behaviour earlier runs had.")
    ap.add_argument("--limit-tasks", type=int, default=None)
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    dataset = construct_kernelbench_dataset(args.level)
    problem_ids = dataset.get_problem_ids()
    if args.limit_tasks:
        problem_ids = problem_ids[:args.limit_tasks]

    os.makedirs(args.out, exist_ok=True)

    # One conversation per (task, sample). Each carries its own message history
    # so a repair turn continues that candidate's own attempt.
    convs = []
    for pid in problem_ids:
        problem = dataset.get_problem_by_id(pid)
        prompt = get_custom_prompt(
            args.prompt_tier, ref_arch_src=problem.code, backend="cutile",
            option="one_shot", precision="fp32")
        for sid in range(args.samples):
            convs.append({
                "key": "%d:%d" % (pid, sid),
                "problem_id": pid,
                "problem": problem.name,
                "ref_src": problem.code,
                "messages": [{"role": "user", "content": prompt}],
                "history": [],          # one entry per round
                "passed_round": None,
                "final_code": None,
            })

    print("tasks %d x %d samples = %d conversations, up to %d repair rounds "
          "(prompt tier %s)"
          % (len(problem_ids), args.samples, len(convs), args.max_rounds,
             args.prompt_tier))

    chat = Chat(args.base_url, args.model, args.temperature, args.top_p,
                args.top_k, args.max_tokens)

    per_round = []
    t_start = time.time()

    with VerifierPool(workers=args.verify_workers, gpus=args.gpus) as pool:
        active = convs
        for rnd in range(args.max_rounds + 1):
            if not active:
                break
            label = "initial" if rnd == 0 else "repair %d" % rnd
            t0 = time.time()

            texts = sample_batch(chat, [c["messages"] for c in active],
                                 args.concurrency)
            t_sample = time.time() - t0

            items, no_code = [], 0
            for c, text in zip(active, texts):
                code = extract_code(text) if not text.startswith("__ERROR__") else None
                c["_last_text"] = text
                c["_last_code"] = code
                if code:
                    items.append((c["key"], code, c["ref_src"]))
                else:
                    no_code += 1

            t1 = time.time()
            results = pool.verify_batch(items)
            t_verify = time.time() - t1

            still_failing = []
            n_pass = 0
            n_inconclusive = 0
            for c in active:
                code = c["_last_code"]
                if code is None:
                    rec = {"stage": "no_code", "error": "no code block in response",
                           "passed": False}
                else:
                    rec = results[c["key"]]
                c["history"].append({
                    "round": rnd, "stage": rec["stage"],
                    "error": (rec.get("error") or "")[:600],
                    "passed": rec["passed"], "code": code,
                })
                if rec["passed"]:
                    c["passed_round"] = rnd
                    c["final_code"] = code
                    n_pass += 1
                elif rec["stage"] in ("oom", "cuda_poison", "worker_crash"):
                    # The harness failed, not the candidate. Telling the model it
                    # ran out of memory would be false feedback, so drop it and
                    # let the analysis exclude it from the denominator.
                    c["inconclusive"] = rec["stage"]
                    n_inconclusive += 1
                elif code is not None:
                    # Continue this candidate's own conversation.
                    c["messages"] = c["messages"] + [
                        {"role": "assistant", "content": "```python\n%s\n```" % code},
                        {"role": "user",
                         "content": build_repair_message(
                             rec["stage"], rec.get("error") or "",
                             bool(rec.get("impure_correct")))},
                    ]
                    still_failing.append(c)
                # A response with no code block gives nothing to repair.

            per_round.append({
                "round": rnd, "attempted": len(active), "passed": n_pass,
                "no_code": no_code, "inconclusive": n_inconclusive,
                "sample_s": round(t_sample, 1), "verify_s": round(t_verify, 1),
            })
            print("  %-9s attempted %4d  passed %4d  no-code %3d  inconclusive %3d "
                  "(sample %.0fs, verify %.0fs)"
                  % (label, len(active), n_pass, no_code, n_inconclusive,
                     t_sample, t_verify))

            active = still_failing

    total_pass = sum(1 for c in convs if c["passed_round"] is not None)
    print("\ntotal: %d/%d candidates eventually passed (%.1f%%) in %.1f min"
          % (total_pass, len(convs), total_pass / len(convs) * 100,
             (time.time() - t_start) / 60))

    # Trajectories: everything, for later agentic training.
    with open(os.path.join(args.out, "trajectories.jsonl"), "w") as f:
        for c in convs:
            f.write(json.dumps({
                "key": c["key"], "problem_id": c["problem_id"],
                "problem": c["problem"], "passed_round": c["passed_round"],
                "inconclusive": c.get("inconclusive"),
                "history": c["history"],
            }) + "\n")

    # Harvested positives: the final corrected kernels, which is what a
    # single-turn SFT round consumes.
    n_written = 0
    for c in convs:
        if c["final_code"] is None:
            continue
        path = os.path.join(
            args.out,
            "level_%d_problem_%d_sample_%d_kernel.py"
            % (args.level, c["problem_id"], int(c["key"].split(":")[1])))
        with open(path, "w") as f:
            f.write(c["final_code"])
        n_written += 1

    with open(os.path.join(args.out, "rounds.json"), "w") as f:
        json.dump({"per_round": per_round, "args": vars(args)}, f, indent=2)

    print("wrote %d harvested kernels and %d trajectories to %s"
          % (n_written, len(convs), args.out))


if __name__ == "__main__":
    main()
