"""Prompt policy for four sequential, verifier-guided candidates."""

from prompts import build_repair_message


ROUND_ROLES = {
    1: """Try a fusion-first implementation. Minimize intermediate tensors and
`ct.launch` calls; use one fused cuTile kernel when that is legal and fast.
Do not sacrifice numerical correctness and do not leave work in PyTorch.""",
    2: """Try a deliberately different decomposition from the supplied code.
Use a stable multi-stage implementation when over-fusion hurts correctness or
occupancy. Change the dataflow/layout rather than only renaming variables.
Every computational operation must still be implemented in cuTile.""",
    3: """Optimize the verified implementation for kernel wall time. Preserve
its mathematics, but change tile sizes, grid mapping, memory traffic, or the
number of launches. Produce a materially different speed candidate, not a
comment-only rewrite. Keep the whole computation in cuTile.""",
}


def failure_priority(entry):
    """Lower is a more useful candidate to repair."""
    result = entry.get("result") or {}
    if result.get("passed"):
        return -1
    stage = result.get("stage") or "no_code"
    if stage == "purity" and result.get("impure_correct"):
        return 0
    if stage == "exec" and result.get("max_diff") is not None:
        return 1
    if stage == "purity":
        return 2
    if stage == "exec":
        return 3
    if stage == "timeout":
        return 4
    if stage == "no_code":
        return 5
    return 6


def choose_seed(history):
    """Choose fastest pass, otherwise the closest/latest failed attempt."""
    passed = [
        row for row in history
        if (row.get("result") or {}).get("passed") and row.get("code")
    ]
    if passed:
        return min(
            passed,
            key=lambda row: (
                float((row.get("result") or {}).get("kernel_ms")
                      or float("inf")),
                -int(row.get("round", 0)),
            ),
        )
    candidates = [row for row in history if row.get("code")]
    if not candidates:
        return history[-1] if history else None
    return min(
        candidates,
        key=lambda row: (failure_priority(row), -int(row.get("round", 0))),
    )


def _code_block(code):
    return "```python\n%s\n```" % (code or "")


def build_adaptive_prompt(base_prompt, history, round_index):
    """Return round 0 unchanged; later rounds include only one prior code."""
    if round_index == 0:
        return base_prompt
    if round_index not in ROUND_ROLES:
        raise ValueError("round must be 0..3, got %r" % round_index)

    role = ROUND_ROLES[round_index]
    seed = choose_seed(history)
    if seed is None or not seed.get("code"):
        followup = """No usable Python code was extracted from the previous
attempt. Start over and produce a complete ModelNew in one Python code block.

%s""" % role
    else:
        result = seed.get("result") or {}
        if result.get("passed"):
            timing = "kernel_ms=%s" % result.get("kernel_ms")
            if result.get("ref_ms") is not None:
                timing += ", torch_compile_ms=%s" % result.get("ref_ms")
            followup = """A previous implementation passed numerical and cuTile
purity verification (%s).

%s

Previous verified implementation:
%s

Output the complete replacement ModelNew in one Python code block. Do not
explain and do not use PyTorch operators for computation.""" % (
                timing, role, _code_block(seed["code"]))
        else:
            repair = build_repair_message(
                result.get("stage") or "no_code",
                result.get("error") or "No verifier diagnostic was produced.",
                bool(result.get("impure_correct")),
            )
            followup = """The previous implementation did not pass.

%s

Previous implementation:
%s

Verifier-guided repair request:
%s""" % (role, _code_block(seed["code"]), repair)

    return base_prompt + "\n\n---\n\n" + followup
