"""Show the size of each prompt tier, and check they all render.

The full prompt is 59k characters of which 55k is the API reference, so what a
tier costs is the main thing to know about it before spending a training run on
it. Rendering here also catches a missing template or context key before a
generation run fails hours in.

Usage:
    python3 scripts/prompt_tiers.py --level 1 --problem-id 1
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--problem-id", type=int, default=1)
    ap.add_argument("--tiers", default="cutile_docs,cutile_concepts,cutile_nodocs")
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    dataset = construct_kernelbench_dataset(args.level)
    problem = dataset.get_problem_by_id(args.problem_id)

    print("level %d problem %d: %s\n" % (args.level, args.problem_id, problem.name))
    print("%-20s %10s %10s %s" % ("tier", "chars", "~tokens", "vs full"))

    full = None
    for tier in args.tiers.split(","):
        prompt = get_custom_prompt(
            tier, ref_arch_src=problem.code, backend="cutile",
            option="one_shot", precision="fp32")
        n = len(prompt)
        if full is None:
            full = n
        # Roughly four characters per token for English prose plus code; exact
        # counts need the tokeniser but the ratio is what matters here.
        print("%-20s %10d %10d %6.1fx smaller"
              % (tier, n, n // 4, full / n))


if __name__ == "__main__":
    main()
