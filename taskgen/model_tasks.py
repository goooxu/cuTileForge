"""Have the model write the tasks, so diversity is not capped by our templates.

The template builders are mechanical and safe, but their diversity is bounded:
600 generated tasks come out as 151 distinct operator shapes because that is all
the builders can express. Beyond that, adding tasks adds shape variants of
patterns the model has already seen.

The model, on the other hand, knows PyTorch extremely well -- PyTorch is not what
it lacks. So it can be asked for realistic modules, which are then validated by
execution the same way template output is. The result is still not external data:
the task is model-written and checked by running it, and the solution is
model-written and checked by the compiler. No human-authored example enters
either side.

Difficulty screening is deliberately not done here. rl/select_frontier.py already
measures per-task pass rates and drops the ones that are always solved or never
solved, and it needs sampled solutions to do that, so it belongs after this.

Usage:
    python3 taskgen/model_tasks.py --level 96 --count 400 --out-root kernelbench/KernelBench
"""

import argparse
import collections
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "repair"))

from generate_tasks import hashes_of_levels, task_hash, validate  # noqa: E402

# Seeds for the request. Without them the model returns the same handful of
# modules over and over; naming a family and a flavour spreads the output. The
# families are weighted toward where the benchmark is hardest rather than toward
# what is easy to ask for.
FAMILIES = [
    "a convolution variant (transposed, dilated, depthwise, grouped, or 1D/3D)",
    "a convolution followed by two or three elementwise operations",
    "a normalisation layer (batch, layer, group, instance, or RMS)",
    "a normalisation followed by a residual add and an activation",
    "a pooling layer (max, average, adaptive, 1D or 3D)",
    "a pooling layer followed by elementwise work",
    "a matrix multiply followed by a bias and an activation",
    "a reduction along one axis followed by elementwise work",
    "a chain of four or more elementwise operations on a large tensor",
    "a small transformer-style block tail: normalise, residual, activation",
    "an attention-style score computation: scale, softmax, then a matmul",
    "a depthwise separable convolution written as two convolutions",
]

FLAVOURS = ["small tensors", "medium tensors", "large tensors suitable for "
            "measuring throughput"]

# Deliberately describes the format in words instead of showing a filled-in
# example. With an example present the model frequently continued it -- replies
# began at the module-level constants with no class at all, which cost three
# quarters of the first runs. Nothing to continue, nothing to get wrong.
TASK_PROMPT = """Write a single self-contained PyTorch file to be used as a
kernel-porting exercise. The computation should be {family}, sized for
{flavour}.

The file must contain, in this order:

1. `import torch` and `import torch.nn as nn`.
2. `class Model(nn.Module)` whose first line is the docstring
   `\"\"\"SomeName (tier {tier}, {category})\"\"\"` -- that exact shape, with a
   descriptive name of your choosing and one of conv, norm, pool, matmul,
   reduction or elementwise as the category. Tooling parses this line.
3. An `__init__(self, ...)` taking whatever configuration it needs, and a
   `forward(self, ...)` returning exactly one tensor.
4. Module-level constants for every shape, as plain assignments.
5. `def get_inputs():` returning a list of tensors to pass to `forward`.
6. `def get_init_inputs():` returning a list of arguments to pass to `__init__`.

Rules: no randomness or dropout in `forward`, no in-place modification of the
inputs, and if the module holds a BatchNorm call `.eval()` on it in `__init__` so
the forward is deterministic. Do not call `.cuda()` or pass `device=` anywhere:
the harness decides placement, and a hardcoded device makes the file untestable
on CPU.

Reply with one fenced Python block containing the whole file and nothing else.
"""


def _validate_worker(src, q):
    """Validate in a fresh process, with the GPU hidden and memory capped."""
    import resource

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    limit = int(os.environ["TASKGEN_MEM_BYTES"])
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        q.put(validate(src))
    except MemoryError:
        q.put("asks for more memory than the cap")
    except BaseException as e:                       # includes SystemExit
        q.put("%s: %s" % (type(e).__name__, str(e)[:120]))


def validate_isolated(src: str, mem_gb: float, timeout_s: float) -> str:
    """validate(), but a pathological task can only kill its own process."""
    import multiprocessing as mp

    os.environ["TASKGEN_MEM_BYTES"] = str(int(mem_gb * 1e9))
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_validate_worker, args=(src, q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return "timed out after %.0fs" % timeout_s
    try:
        return q.get_nowait()
    except Exception:
        # Died without reporting: killed by the allocator or a hard crash.
        return "crashed during validation"


def extract_task(text: str) -> str:
    """Pull out the fenced block that defines a task.

    Not kernelbench's extract_best_code: that one is tuned for solutions and
    prefers the last `class ModelNew` block, so on a task it falls through its
    heuristics and returns something without a `Model` in it. That accounted for
    290 of 316 validation failures on the first run.
    """
    blocks = re.findall(r"```+(?:python)?[ \t]*\n(.*?)```+", text, re.DOTALL)
    wanted = [b for b in blocks if re.search(r"class\s+Model\s*\(", b)]
    if wanted:
        # The last one, in case the reply revises itself.
        return wanted[-1].strip()

    # Unfenced, or fenced in a way the pattern missed. Take from the first import
    # to the end; validation will reject it if that was the wrong guess.
    if re.search(r"class\s+Model\s*\(", text):
        start = text.find("import ")
        return text[start if start >= 0 else 0:].strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out-root", default=os.path.join(HERE, "..", "kernelbench",
                                                       "KernelBench"))
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--exclude-levels", default=None,
                    help="Comma-separated levels whose tasks must not recur here.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.1,
                    help="Higher than for solutions: here variety is the point.")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--oversample", type=float, default=1.8,
                    help="Requests per wanted task, to absorb invalid output.")
    ap.add_argument("--mem-limit-gb", type=float, default=32.0,
                    help="Address-space cap for validating one task. RLIMIT_AS "
                         "counts reserved virtual memory, and torch reserves a "
                         "lot, so this has to be far above the actual tensors; "
                         "at 8 GB even importing torch failed. The cases worth "
                         "catching asked for 128 GiB.")
    ap.add_argument("--validate-timeout", type=float, default=60.0)
    args = ap.parse_args()

    import random

    from repair_loop import Chat, sample_batch

    rng = random.Random(args.seed)
    out_dir = os.path.join(args.out_root, "level%d" % args.level)
    os.makedirs(out_dir, exist_ok=True)
    if args.clean:
        for f in os.listdir(out_dir):
            if f.endswith(".py"):
                os.remove(os.path.join(out_dir, f))

    n_req = int(args.count * args.oversample)
    prompts = []
    for _ in range(n_req):
        tier = rng.choice([2, 3, 5])
        prompts.append([{"role": "user", "content": TASK_PROMPT.format(
            family=rng.choice(FAMILIES),
            flavour=rng.choice(FLAVOURS),
            name_hint="DescriptiveName",
            tier=tier,
            category="conv",
        )}])

    print("asking for %d modules to keep %d" % (n_req, args.count))
    chat = Chat(args.base_url, args.model, args.temperature, 0.95, 50, 2048)
    texts = sample_batch(chat, prompts, args.concurrency)

    # Seeding the dedup set with the training levels' hashes is what makes a
    # held-out set actually held out.
    seen = set()
    if args.exclude_levels:
        seen = hashes_of_levels(
            args.out_root, [int(x) for x in args.exclude_levels.split(",")])
        print("excluding %d task hashes from levels %s"
              % (len(seen), args.exclude_levels))
    written = 0
    # API failures and unparseable replies need separate counters: conflating them
    # once hid the fact that most of a run's losses were the server erroring under
    # concurrency rather than the model writing bad tasks.
    n_apierr = n_nocode = n_badformat = n_invalid = n_dup = 0

    # Model-written tasks have no bound on the shapes they ask for: one run tried
    # to allocate 128 GiB and took the GPU down with it. So validation runs in a
    # subprocess with an address-space cap and a timeout, and a task that trips
    # either is simply rejected.
    def check(src):
        return validate_isolated(src, mem_gb=args.mem_limit_gb,
                                 timeout_s=args.validate_timeout)

    candidates, api_errors = [], []
    for text in texts:
        if text.startswith("__ERROR__"):
            n_apierr += 1
            api_errors.append(" ".join(text.split()[1:5]))
            continue
        src = extract_task(text)
        if not src:
            n_nocode += 1
            continue
        if not re.search(r'"""[\w ]+ \(tier \d+, (conv|norm|pool|matmul|reduction'
                         r'|elementwise)\)', src):
            n_badformat += 1
            continue
        h = task_hash(src)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        candidates.append(src)

    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = list(ex.map(check, candidates))

    fail_reasons = []
    for src, err in zip(candidates, verdicts):
        if written >= args.count:
            break
        if err:
            n_invalid += 1
            # Keep the exception type and the first few words: the whole message
            # carries tensor shapes and would never group.
            fail_reasons.append(" ".join(err.split()[:6]))
            continue
        m = re.search(r'"""([\w ]+) \(tier (\d+), (\w+)\)', src)
        name = re.sub(r"\W+", "", m.group(1))[:40] or "Model"
        path = os.path.join(out_dir, "%d_%s_t%s.py" % (written + 1, name, m.group(2)))
        with open(path, "w") as f:
            f.write(src)
        written += 1

    print("wrote %d problems to %s" % (written, out_dir))
    print("  discarded: %d API errors, %d unparseable, %d bad docstring, "
          "%d failed to run, %d duplicate"
          % (n_apierr, n_nocode, n_badformat, n_invalid, n_dup))
    if api_errors:
        for reason, n in collections.Counter(api_errors).most_common(3):
            print("    %4d  %s" % (n, reason))
    if fail_reasons:
        # A high rejection rate is usually one repeated mistake in the request
        # rather than the model being incapable, so name the top offenders.
        print("  why they failed to run:")
        for reason, n in collections.Counter(fail_reasons).most_common(6):
            print("    %4d  %s" % (n, reason))
    if written < args.count:
        print("  short of the target; raise --oversample")


if __name__ == "__main__":
    main()
