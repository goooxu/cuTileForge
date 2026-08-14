"""Prompt text for the multi-turn repair loop.

This lives here rather than in KernelBench's prompts.toml because the repair
loop is an addition of this project, not a KernelBench feature; putting it in
the upstream file would grow patches/0001-cutile-backend.patch for something
upstream has no use for.

The repair turn is sent as a new user message in the same conversation, so the
documentation bundle and the task description are already in context and are not
repeated. That matters for cost: the bundle is ~14k tokens and prefix caching
makes the continuation almost free.
"""

# cuTile diagnostics carry file, line and column, e.g.
#   Invalid argument "shape" of load(): Expected shape length to be 3, got 2
#     "/tmp/tmpXXXX.py", line 38, col 19-57, in bmm_kernel:
# so the message is passed through verbatim rather than summarised.
REPAIR_TURN = """Your kernel was rejected.

{failure_kind}
{error_text}

Fix it and output the complete corrected ModelNew in a single Python code block.
Do not explain. Do not fall back to PyTorch operators: the entire computation
must stay in cuTile."""


# How each verifier outcome is described to the model. The wording is meant to
# point at the class of mistake without telling it the fix, so the loop measures
# the model's ability to act on a compiler diagnostic rather than on a hint.
FAILURE_KIND = {
    "purity": "It did not implement the computation entirely in cuTile:",
    "exec": "It failed when compiled or run:",
    "timeout": "It did not finish in time, which usually means an unbounded loop:",
    "oom": "It ran out of GPU memory:",
}

# A purity failure whose numbers were verified correct is a different situation
# from every other failure, and worth saying so. It is also the largest one left:
# on the benchmark's Level 2, roughly three quarters of what the best model cannot
# solve is a kernel that computes the right answer and gets rejected for leaving a
# pointwise activation in torch.
#
# The generic purity wording does not mention that the numerics passed, so the
# model has no reason to believe its kernel is otherwise sound and tends to
# rewrite the whole thing -- which risks losing the part that already worked. This
# says what is actually needed: keep everything, port the named ops.
PURITY_CORRECT = """Your kernel produced the correct numbers, so the computation
itself is right. It was rejected only because part of the work is still done by
PyTorch rather than cuTile:

{error_text}

Port the remaining operation into the cuTile kernel and change nothing else.
Output the complete ModelNew in a single Python code block. Do not explain."""


def build_repair_message(stage: str, error: str,
                         numerically_correct: bool = False) -> str:
    if stage == "purity" and numerically_correct:
        return PURITY_CORRECT.format(error_text=error.strip())
    kind = FAILURE_KIND.get(stage, "It failed:")
    return REPAIR_TURN.format(failure_kind=kind, error_text=error.strip())
