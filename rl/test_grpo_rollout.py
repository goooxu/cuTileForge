#!/usr/bin/env python3
"""The three GRPO blockers that would train Glimmer on the wrong tokens.

1. Chat used to return empty when the server put the trace in reasoning_content.
2. Chat used to leave skip_special_tokens at the server default, which strips
   Glimmer's channel markers and makes every sample look truncated.
3. The trainer used to score extract_code(text) and front-truncate overflows,
   so the gradient never touched the reasoning tokens the policy sampled.
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "repair"))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))

from repair_loop import chat_extra_body, choice_text  # noqa: E402
from rollout_tokens import build_sequence  # noqa: E402
from select_frontier import HELDOUT_LEVELS  # noqa: E402


class FakeTok:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True, **kw):
        text = messages[0]["content"]
        if kw.get("reasoning_strength"):
            text = "<%s>" % kw["reasoning_strength"] + text
        if add_generation_prompt:
            text += "<gen>"
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def check(cond, msg, fails):
    print("  %-4s %s" % ("ok" if cond else "FAIL", msg))
    if not cond:
        fails.append(msg)


def main() -> None:
    fails = []

    print("choice_text:")
    empty = SimpleNamespace(message=SimpleNamespace(
        content="", reasoning_content="to=self think <|eom|> ```python\nk=1```"))
    check(choice_text(empty).startswith("to=self"),
          "falls back to reasoning_content when content is empty", fails)
    listed = SimpleNamespace(message=SimpleNamespace(
        content=[{"text": "hello "}, {"text": "world"}],
        reasoning_content="ignored"))
    check(choice_text(listed) == "hello world",
          "joins list content and ignores the reasoning field", fails)
    both = SimpleNamespace(message=SimpleNamespace(
        content="final answer", reasoning_content="secret"))
    check(choice_text(both) == "final answer",
          "prefers content when it is non-empty", fails)

    print("\nchat_extra_body:")
    saved = {k: os.environ.get(k) for k in (
        "ENABLE_THINKING", "REASONING_EFFORT", "REASONING_STRENGTH",
        "KEEP_SPECIAL_TOKENS")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ["REASONING_STRENGTH"] = "xhigh"
        os.environ["KEEP_SPECIAL_TOKENS"] = "1"
        extra = chat_extra_body(40)
        check(extra.get("top_k") == 40, "forwards top_k", fails)
        check(extra.get("skip_special_tokens") is False,
              "KEEP_SPECIAL_TOKENS keeps channel markers", fails)
        check((extra.get("chat_template_kwargs") or {}).get("reasoning_strength")
              == "xhigh",
              "REASONING_STRENGTH goes into chat_template_kwargs", fails)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print("\nbuild_sequence:")
    tok = FakeTok()
    sampled = "to=self long reasoning trace\n```python\nreturn x\n```"
    ids, mask = build_sequence(tok, "task", sampled, 20480, turn_end="<|eot|>")
    n_prompt = mask.index(1)
    n_comp = sum(mask)
    check(ids is not None and n_comp == len(sampled) + len("<|eot|>"),
          "loss mask covers the sampled text plus the turn end (%d tokens)"
          % n_comp, fails)
    check(n_comp > len("return x"),
          "mask is not the extracted kernel (%d > %d)"
          % (n_comp, len("return x")), fails)
    check(all(m == 0 for m in mask[:n_prompt])
          and all(m == 1 for m in mask[n_prompt:]),
          "prompt positions are 0, completion positions are 1", fails)

    dropped, dropped_mask = build_sequence(
        tok, "task", sampled, max_len=8, turn_end="<|eot|>")
    check(dropped is None and dropped_mask is None,
          "overflow drops rather than front-truncating the prompt", fails)

    a, _ = build_sequence(tok, "task", "x", 20480,
                          chat_kwargs={"reasoning_strength": "xhigh"},
                          turn_end="<|eot|>")
    b, _ = build_sequence(tok, "task", "x", 20480, turn_end="<|eot|>")
    check(len(a) > len(b),
          "reasoning_strength changes the prompt prefix", fails)

    print("\nheld-out guard:")
    check(HELDOUT_LEVELS == frozenset({60, 84, 88, 97, 98, 99}),
          "select_frontier refuses eval/held-out levels", fails)
    grpo_src = open(os.path.join(HERE, "grpo.py")).read()
    check("HELDOUT_LEVELS = frozenset({60, 84, 88, 97, 98, 99})" in grpo_src,
          "grpo.py uses the same held-out set", fails)
    check("pool=pool" in grpo_src and "VerifierPool" in grpo_src,
          "grpo.py keeps one VerifierPool for the whole run", fails)

    print("\nscore_rollouts pool reuse:")
    from reward import score_rollouts  # noqa: E402

    class FakePool:
        def __init__(self):
            self.n = 0

        def verify_batch(self, items):
            self.n += 1
            return {k: {"passed": True, "stage": "pass", "speedup": None}
                    for k, _, _ in items}

    fake = FakePool()
    scored = score_rollouts(
        [("0", "code", "ref"), ("1", None, "ref")],
        measure_speed=False, pool=fake)
    check(fake.n == 1, "uses the passed pool instead of opening another", fails)
    check(scored["0"][0] == 1.0, "passed kernel scores 1.0", fails)
    check(scored["1"][0] == 0.0, "missing code scores 0 without the verifier", fails)
    score_rollouts([("2", "code", "ref")], measure_speed=False, pool=fake)
    check(fake.n == 2, "the same pool can score a later iteration", fails)

    print("\n%s" % ("all checks passed" if not fails else "%d FAILED" % len(fails)))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
