#!/usr/bin/env python3
"""State engine for verifier-guided 1+1+1+1 kernel search.

Generation and GPU verification are deliberately separate subcommands.  The
host orchestrator starts vLLM for ``generate``, stops it, runs fast_verify in a
fresh exclusive-GPU container, then calls ``update`` with that round's JSONL.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sequential_prompts import build_adaptive_prompt  # noqa: E402


NO_CODE_STUB = (
    "# sequential search could not extract a complete ModelNew code block\n"
)


def _choice_text(choice):
    from repair_loop import choice_text
    return choice_text(choice)


def _chat_extra_body(top_k):
    from repair_loop import chat_extra_body
    return chat_extra_body(top_k)


class DynamicChat:
    """OpenAI chat client with a per-prompt native-context output budget."""

    def __init__(self, base_url, model, tokenizer_path, temperature, top_p,
                 top_k, desired_max_tokens, native_context, safety_margin):
        from openai import OpenAI
        from transformers import AutoTokenizer

        self.client = OpenAI(
            api_key="local-no-auth", base_url=base_url,
            timeout=3600, max_retries=2)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.desired_max_tokens = int(desired_max_tokens)
        self.native_context = int(native_context)
        self.safety_margin = int(safety_margin)
        self.extra_body = _chat_extra_body(top_k)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True)

    def estimate_input_tokens(self, messages):
        template_kwargs = {}
        thinking = os.environ.get("ENABLE_THINKING", "").strip().lower()
        if thinking in ("0", "1", "true", "false", "yes", "no"):
            template_kwargs["enable_thinking"] = thinking in (
                "1", "true", "yes")
        strength = os.environ.get("REASONING_STRENGTH", "").strip()
        if strength:
            template_kwargs["reasoning_strength"] = strength
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                **template_kwargs)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids.get("input_ids", ids)
            if hasattr(ids, "numel"):
                return int(ids.numel())
            if (isinstance(ids, (list, tuple)) and ids
                    and isinstance(ids[0], (list, tuple))):
                return len(ids[0])
            return len(ids)
        except Exception:
            # Conservative fallback for family-specific templates.
            text = "\n".join(str(row.get("content") or "") for row in messages)
            return len(self.tokenizer.encode(text)) + 64

    def _request(self, messages, input_tokens, max_tokens):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return {
            "text": _choice_text(choice),
            "input_tokens_estimate": input_tokens,
            "max_tokens_sent": max_tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "desired_max_tokens": self.desired_max_tokens,
            "native_context": self.native_context,
            "safety_margin": self.safety_margin,
            "hit_token_limit": getattr(choice, "finish_reason", None) == "length",
            "error": None,
        }

    def complete(self, messages):
        input_tokens = self.estimate_input_tokens(messages)
        available = self.native_context - input_tokens - self.safety_margin
        if available <= 0:
            return {
                "text": "",
                "input_tokens_estimate": input_tokens,
                "max_tokens_sent": 0,
                "finish_reason": "context_exhausted",
                "prompt_tokens": None,
                "completion_tokens": None,
                "desired_max_tokens": self.desired_max_tokens,
                "native_context": self.native_context,
                "safety_margin": self.safety_margin,
                "hit_token_limit": True,
                "error": "prompt leaves no output tokens in native context",
            }
        max_tokens = min(self.desired_max_tokens, available)
        try:
            return self._request(messages, input_tokens, max_tokens)
        except Exception as error:
            # vLLM's 400 response reports the server-side prompt count. Retry
            # once with that authoritative value if tokenizer estimates differ.
            match = re.search(
                r"prompt contains at least (\d+) input tokens", str(error))
            if match is None:
                match = re.search(
                    r"parameter=input_tokens,\s*value=(\d+)", str(error))
            if match:
                actual = int(match.group(1))
                retry_max = min(
                    self.desired_max_tokens,
                    self.native_context - actual - self.safety_margin)
                if 0 < retry_max < max_tokens:
                    try:
                        row = self._request(messages, actual, retry_max)
                        row["input_tokens_estimate"] = input_tokens
                        row["prompt_tokens_from_error"] = actual
                        return row
                    except Exception as retry_error:
                        error = retry_error
            return {
                "text": "",
                "input_tokens_estimate": input_tokens,
                "max_tokens_sent": max_tokens,
                "finish_reason": "error",
                "prompt_tokens": None,
                "completion_tokens": None,
                "desired_max_tokens": self.desired_max_tokens,
                "native_context": self.native_context,
                "safety_margin": self.safety_margin,
                "hit_token_limit": False,
                "error": "%s: %s" % (
                    type(error).__name__, str(error)[:1000]),
            }


def sample_dynamic_batch(chat, message_lists, concurrency, on_result=None):
    out = [None] * len(message_lists)

    def one(index):
        return index, chat.complete(message_lists[index])

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(one, index): index
            for index in range(len(message_lists))
        }
        for future in as_completed(futures):
            index, row = future.result()
            out[index] = row
            if on_result is not None:
                on_result(index, row)
    return out


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp), str(path))


def load_state(run_dir):
    path = Path(run_dir) / "state.json"
    if not path.is_file():
        raise SystemExit("missing state: %s; run init first" % path)
    return path, json.loads(path.read_text())


def task_rows(state):
    return [state["tasks"][key] for key in sorted(
        state["tasks"], key=lambda value: int(value))]


def round_dir(run_dir, round_index):
    return Path(run_dir) / ("round_%d" % round_index)


def kernel_path(run_dir, level, pid, round_index):
    return round_dir(run_dir, round_index) / (
        "level_%d_problem_%d_sample_%d_kernel.py"
        % (level, pid, round_index))


def response_path(run_dir, level, pid, round_index):
    return round_dir(run_dir, round_index) / (
        "level_%d_problem_%d_sample_%d_response.txt"
        % (level, pid, round_index))


def prompt_path(run_dir, level, pid, round_index):
    return round_dir(run_dir, round_index) / (
        "level_%d_problem_%d_sample_%d_prompt.txt"
        % (level, pid, round_index))


def metadata_path(run_dir, level, pid, sample_id, directory=None):
    root = Path(directory) if directory else round_dir(run_dir, sample_id)
    return root / (
        "level_%d_problem_%d_sample_%d_meta.json"
        % (level, pid, sample_id))


def select_problem_ids(dataset, ids, limit):
    available = list(dataset.get_problem_ids())
    if ids:
        wanted = [int(value) for value in ids.split(",") if value.strip()]
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise SystemExit("problem ids absent from level: %s" % missing)
        available = [pid for pid in available if pid in set(wanted)]
    if limit:
        available = available[:limit]
    return available


def command_init(args):
    from kernelbench.dataset import construct_kernelbench_dataset

    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if int(state["level"]) != args.level or state["tag"] != args.tag:
            raise SystemExit("existing state belongs to %s level %s"
                             % (state["tag"], state["level"]))
        print("state already initialized: %s (%d tasks)"
              % (state_path, len(state["tasks"])))
        return

    dataset = construct_kernelbench_dataset(args.level)
    pids = select_problem_ids(dataset, args.ids, args.limit)
    if not pids:
        raise SystemExit("no problems selected")
    tasks = {}
    for pid in pids:
        problem = dataset.get_problem_by_id(pid)
        tasks[str(pid)] = {
            "problem_id": int(pid),
            "problem": problem.name,
            "history": [],
        }
    state = {
        "version": 1,
        "tag": args.tag,
        "level": args.level,
        "prompt_tier": args.prompt_tier,
        "rounds": 4,
        "tasks": tasks,
    }
    atomic_json(state_path, state)
    print("initialized %s: %s level %d, %d tasks"
          % (state_path, args.tag, args.level, len(tasks)))


def history_before(task, round_index):
    return [
        row for row in task.get("history", [])
        if int(row.get("round", -1)) < round_index
    ]


def command_generate(args):
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt
    from kernelbench.utils import extract_best_code

    state_path, state = load_state(args.run_dir)
    if not 0 <= args.round <= 3:
        raise SystemExit("--round must be 0..3")
    if args.round > 0:
        incomplete = [
            row["problem_id"] for row in task_rows(state)
            if not any(int(item.get("round", -1)) == args.round - 1
                       for item in row.get("history", []))
        ]
        if incomplete:
            raise SystemExit(
                "round %d feedback missing for %d tasks, e.g. %s"
                % (args.round - 1, len(incomplete), incomplete[:4]))

    dataset = construct_kernelbench_dataset(int(state["level"]))
    out_dir = round_dir(args.run_dir, args.round)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    messages = []
    prompts = []
    for task in task_rows(state):
        pid = int(task["problem_id"])
        kpath = kernel_path(args.run_dir, state["level"], pid, args.round)
        if kpath.is_file():
            continue
        problem = dataset.get_problem_by_id(pid)
        base_prompt = get_custom_prompt(
            state["prompt_tier"],
            ref_arch_src=problem.code,
            backend="cutile",
            option="one_shot",
            precision="fp32",
        )
        prompt = build_adaptive_prompt(
            base_prompt, history_before(task, args.round), args.round)
        pending.append(task)
        messages.append([{"role": "user", "content": prompt}])
        prompts.append(prompt)

    if not pending:
        print("round %d already generated" % args.round)
        return

    chat = DynamicChat(
        args.base_url, args.model, args.tokenizer,
        args.temperature, args.top_p, args.top_k,
        args.max_tokens, args.native_context, args.safety_margin)
    counts = {"code": 0, "error": 0}

    def persist(index, completion):
        task = pending[index]
        prompt = prompts[index]
        pid = int(task["problem_id"])
        text = completion.get("text") or ""
        request_meta = {
            key: value for key, value in completion.items() if key != "text"
        }
        metadata_path(
            args.run_dir, state["level"], pid, args.round).write_text(
                json.dumps(request_meta, indent=2, sort_keys=True) + "\n")
        if completion.get("error"):
            response_path(
                args.run_dir, state["level"], pid, args.round).write_text(
                    "__ERROR__ " + completion["error"])
            counts["error"] += 1
            return
        code = None
        if text:
            code = extract_best_code(text, ["python", "cpp"])
        prompt_path(args.run_dir, state["level"], pid, args.round).write_text(
            prompt)
        response_path(args.run_dir, state["level"], pid, args.round).write_text(
            text or "")
        kernel_path(args.run_dir, state["level"], pid, args.round).write_text(
            code if code is not None else NO_CODE_STUB)
        counts["code"] += int(code is not None)

    sample_dynamic_batch(
        chat, messages, args.concurrency, on_result=persist)

    marker = {
        "round": args.round,
        "attempted": len(pending),
        "extracted_code": counts["code"],
        "api_errors": counts["error"],
        "total_tasks": len(state["tasks"]),
    }
    atomic_json(out_dir / "generated.json", marker)
    print("round %d attempted %d tasks, extracted %d code blocks, %d API errors"
          % (args.round, len(pending), counts["code"], counts["error"]))


def command_control(args):
    """Generate independent same-prompt candidates with dynamic budgets."""
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt
    from kernelbench.utils import extract_best_code

    dataset = construct_kernelbench_dataset(args.level)
    pids = select_problem_ids(dataset, args.ids, args.limit)
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    messages = []
    prompts = []
    for pid in pids:
        problem = dataset.get_problem_by_id(pid)
        base_prompt = get_custom_prompt(
            args.prompt_tier,
            ref_arch_src=problem.code,
            backend="cutile",
            option="one_shot",
            precision="fp32",
        )
        for sample_id in range(args.samples):
            kpath = out_dir / (
                "level_%d_problem_%d_sample_%d_kernel.py"
                % (args.level, pid, sample_id))
            if kpath.is_file():
                continue
            pending.append((int(pid), sample_id))
            messages.append([{"role": "user", "content": base_prompt}])
            prompts.append(base_prompt)

    if not pending:
        print("control already generated: %d tasks x %d"
              % (len(pids), args.samples))
        return

    chat = DynamicChat(
        args.base_url, args.model, args.tokenizer,
        args.temperature, args.top_p, args.top_k,
        args.max_tokens, args.native_context, args.safety_margin)
    counts = {"code": 0, "error": 0}

    def persist(index, completion):
        pid, sample_id = pending[index]
        prompt = prompts[index]
        prefix = "level_%d_problem_%d_sample_%d" % (
            args.level, pid, sample_id)
        text = completion.get("text") or ""
        request_meta = {
            key: value for key, value in completion.items() if key != "text"
        }
        (out_dir / (prefix + "_meta.json")).write_text(
            json.dumps(request_meta, indent=2, sort_keys=True) + "\n")
        (out_dir / (prefix + "_prompt.txt")).write_text(prompt)
        if completion.get("error"):
            (out_dir / (prefix + "_response.txt")).write_text(
                "__ERROR__ " + completion["error"])
            counts["error"] += 1
            return
        code = extract_best_code(text, ["python", "cpp"]) if text else None
        (out_dir / (prefix + "_response.txt")).write_text(text)
        (out_dir / (prefix + "_kernel.py")).write_text(
            code if code is not None else NO_CODE_STUB)
        counts["code"] += int(code is not None)

    sample_dynamic_batch(
        chat, messages, args.concurrency, on_result=persist)

    atomic_json(out_dir / "generated.json", {
        "attempted": len(pending),
        "extracted_code": counts["code"],
        "api_errors": counts["error"],
        "tasks": len(pids),
        "samples": args.samples,
    })
    print("control attempted %d, extracted %d code blocks, %d API errors"
          % (len(pending), counts["code"], counts["error"]))


def load_verified(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["key"])] = row
    return rows


def write_trajectories(run_dir, state):
    path = Path(run_dir) / "trajectories.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for task in task_rows(state):
            f.write(json.dumps({
                "tag": state["tag"],
                "level": state["level"],
                "problem_id": task["problem_id"],
                "problem": task["problem"],
                "history": task.get("history", []),
            }) + "\n")
    os.replace(str(tmp), str(path))


def command_update(args):
    state_path, state = load_state(args.run_dir)
    verified = load_verified(args.verified)
    missing = []
    for task in task_rows(state):
        pid = int(task["problem_id"])
        key = "%d:%d" % (pid, args.round)
        if key not in verified:
            missing.append(key)
            continue
        kpath = kernel_path(args.run_dir, state["level"], pid, args.round)
        code = kpath.read_text() if kpath.is_file() else None
        if code == NO_CODE_STUB:
            code = None
        mpath = metadata_path(
            args.run_dir, state["level"], pid, args.round)
        request_meta = json.loads(mpath.read_text()) if mpath.is_file() else {}
        entry = {
            "round": args.round,
            "code": code,
            "result": verified[key],
            "request": request_meta,
            "kernel_file": str(kpath),
            "response_file": str(response_path(
                args.run_dir, state["level"], pid, args.round)),
        }
        history = [
            row for row in task.get("history", [])
            if int(row.get("round", -1)) != args.round
        ]
        history.append(entry)
        task["history"] = sorted(
            history, key=lambda row: int(row["round"]))
    if missing:
        raise SystemExit("verified results missing %d keys, e.g. %s"
                         % (len(missing), missing[:4]))
    atomic_json(state_path, state)
    write_trajectories(args.run_dir, state)
    atomic_json(round_dir(args.run_dir, args.round) / "updated.json", {
        "round": args.round,
        "verified": args.verified,
        "tasks": len(state["tasks"]),
    })
    print("round %d feedback stored for %d tasks"
          % (args.round, len(state["tasks"])))


def command_status(args):
    _, state = load_state(args.run_dir)
    counts = {}
    for round_index in range(4):
        counts[round_index] = sum(
            any(int(row.get("round", -1)) == round_index
                for row in task.get("history", []))
            for task in task_rows(state)
        )
    print("tag %s level %s tasks %d rounds %s"
          % (state["tag"], state["level"], len(state["tasks"]), counts))


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")

    init = sub.add_parser("init")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--tag", required=True)
    init.add_argument("--level", required=True, type=int)
    init.add_argument("--prompt-tier", default="cutile_concepts")
    init.add_argument("--ids", default=None)
    init.add_argument("--limit", type=int, default=None)
    init.set_defaults(func=command_init)

    generate = sub.add_parser("generate")
    generate.add_argument("--run-dir", required=True)
    generate.add_argument("--round", required=True, type=int)
    generate.add_argument("--base-url", default="http://localhost:8000/v1")
    generate.add_argument("--model", default="Qwen3-Coder-Next")
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--native-context", required=True, type=int)
    generate.add_argument("--safety-margin", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--max-tokens", type=int, default=32768)
    generate.add_argument("--concurrency", type=int, default=32)
    generate.set_defaults(func=command_generate)

    control = sub.add_parser("control")
    control.add_argument("--run-dir", required=True)
    control.add_argument("--level", required=True, type=int)
    control.add_argument("--prompt-tier", default="cutile_concepts")
    control.add_argument("--ids", default=None)
    control.add_argument("--limit", type=int, default=None)
    control.add_argument("--samples", type=int, default=4)
    control.add_argument("--base-url", default="http://localhost:8000/v1")
    control.add_argument("--model", default="Qwen3-Coder-Next")
    control.add_argument("--tokenizer", required=True)
    control.add_argument("--native-context", required=True, type=int)
    control.add_argument("--safety-margin", type=int, default=256)
    control.add_argument("--temperature", type=float, default=1.0)
    control.add_argument("--top-p", type=float, default=0.95)
    control.add_argument("--top-k", type=int, default=40)
    control.add_argument("--max-tokens", type=int, default=131072)
    control.add_argument("--concurrency", type=int, default=32)
    control.set_defaults(func=command_control)

    update = sub.add_parser("update")
    update.add_argument("--run-dir", required=True)
    update.add_argument("--round", required=True, type=int)
    update.add_argument("--verified", required=True)
    update.set_defaults(func=command_update)

    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if not getattr(args, "command", None):
        ap.error("choose init, generate, control, update, or status")
    args.func(args)


if __name__ == "__main__":
    main()
