# Prompt tier gate: how much documentation does the model actually need?

model-E (phase 6), Level 1, k=4, no training on reduced-doc prompts. The question
is whether training can be moved off the 15k-token prompt, since the API
reference is 91% of every training sequence and that is what has capped every
round at ~500 examples.

| tier              | prompt tokens | pass@1 | entirely cuTile |
| ----------------- | ------------: | -----: | --------------: |
| cutile_docs       |        14,891 |  23.2% |               - |
| cutile_concepts   |         2,393 |  18.0% |           78.5% |
| cutile_nodocs     |           939 |   5.5% |           84.8% |

Verdict: take the concepts tier. It keeps 78% of the pass rate for 16% of the
prompt, a 6.2x throughput multiplier, and it does so before any training on
reduced-doc prompts at all.

nodocs is too low to start from at 5.5%, but it is not zero, and its purity is
the *highest* of the three: without documentation the model still writes
cuTile-shaped code rather than falling back to PyTorch. It remembers what the DSL
looks like and is missing the details. So a fraction of nodocs examples goes into
the training mix rather than being abandoned -- writing the DSL from memory is
the actual definition of the goal.

Numbers by tier are not comparable to each other or to earlier rounds, so every
run records the tier it used (PROMPT_TIER in scripts/run_generate.sh).
