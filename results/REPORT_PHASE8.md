# Phase 8: prompt tiers, and what the model learned to do without the manual

Every round through phase 7 trained on a 15k-token prompt of which the cuTile API
reference was 91%, and the completion being learned from was 3.1%. That is the
mechanical reason no round ever trained on more than 642 examples or ran more
than 44 optimiser steps. It is also the conceptual reason the model could not
become expert at the DSL: the manual was always open in front of it.

Numbers are not comparable across prompt tiers, so the tier is stated everywhere.

## Held-out dev set (KernelBench Level 1+2, 200 problems, k=4)

Criterion throughout: numerically correct AND entirely cuTile, official harness.

| model                        | prompt tokens | pass@1 | pass@4 | solved  | fast_1.0 |
| ---------------------------- | ------------: | -----: | -----: | ------: | -------: |
| baseline, full docs          |        14,891 |  12.9% |  23.5% |  47/200 |    5/200 |
| E (phase 6), full docs       |        14,891 |  14.2% |  25.5% |  51/200 |   11/200 |
| E + repair loop (phase 7)    |        14,891 |  21.6% |  38.0% |  76/200 |   17/200 |
| **F (phase 8), concepts**    |     **2,393** | **22.0%** | **42.0%** | **84/200** | **17/200** |
| **F (phase 8), no docs**     |       **939** |  20.2% |  38.5% |  77/200 | **19/200** |

F single-shot at 2,393 tokens beats E with the full 14,891-token reference and
three rounds of compile feedback, which cost 3.3x the model calls.

F with no documentation at all solves 77 problems. Every earlier model, holding
the complete 55k-character API reference in context, solved at most 51.

### By category, F at the concepts tier against baseline

| category   | problems | baseline    | F           |
| ---------- | -------: | ----------- | ----------- |
| conv       |       98 | 5, fast 0   | **33, fast 9** |
| norm       |       24 | 2, fast 0   | 11, fast 3  |
| activation |       29 | 19, fast 3  | 22, fast 3  |
| matmul     |       15 | 11, fast 1  | 10, fast 1  |
| reduction  |       11 | 1, fast 0   | 2, fast 0   |
| pool       |       10 | 0, fast 0   | 0, fast 0   |
| loss       |        6 | 2, fast 1   | 1, fast 1   |

Convolution is half the benchmark and goes from 5 solved to 33. Pooling is still
zero: 10 problems, and the training pool only ever yielded 115 pooling kernels
because pooling has the lowest pass rate of any family. Loss functions regress
again, as in every round that gives them no data.

### Level 2 on its own

Level 2 is entirely operator chains and is the half nothing had moved: 16 problems
at baseline, 15 after six rounds.

| model            | pass@1 | pass@4 | solved  | fast_1.0 |
| ---------------- | -----: | -----: | ------: | -------: |
| baseline         |   5.2% |  16.0% |  16/100 |    0/100 |
| E (phase 6)      |   5.2% |  15.0% |  15/100 |    4/100 |
| **F, concepts**  | **14.2%** | **38.0%** | **38/100** | **8/100** |
| F, no docs       |  12.0% |  32.0% |  32/100 |    9/100 |

Chains were added to the task generator this round for exactly this reason, since
no earlier task had Level 2's shape.

## Final test (KernelBench Level 3, 50 whole-network problems)

Sealed since phase 6 and opened once, here. These are ResNet, VGG and DenseNet
architectures, not single operators, and nothing was tuned against them.

| model                     | pass@1 | fast_1.0 |
| ------------------------- | -----: | -------: |
| baseline, full docs       |   5.0% |     2/50 |
| **F, concepts tier**      | **20.0%** | **4/50** |
| F, no docs                |  11.5% |     0/50 |

Four times the baseline pass rate on a test set that informed no decision.

## The prompt tier gate

Measured on E before any training on reduced prompts, Level 1:

| tier            | prompt tokens | pass@1 | entirely cuTile |
| --------------- | ------------: | -----: | --------------: |
| cutile_docs     |        14,891 |  23.2% |               - |
| cutile_concepts |         2,393 |  18.0% |           78.5% |
| cutile_nodocs   |           939 |   5.5% |           84.8% |

The concepts tier keeps 78% of the pass rate for 16% of the prompt, so it became
the training tier. nodocs at 5.5% was too low to train on alone but was not zero,
and its purity was the highest of the three -- without documentation the model
still wrote cuTile-shaped code rather than falling back to PyTorch. So 15% of the
training set was rendered with no documentation, and that is what produced a model
that works at 939 tokens.

## Scale, and the side effect nobody predicted

| | phases 2-7 | phase 8 |
| --- | ---: | ---: |
| task definitions | 1,480 total | 3,200 |
| distinct operator shapes | 29 builders | 651 |
| training examples | 349-642 | 2,443 |
| optimiser steps | 44 | 306 |
| tokens per epoch | 5.6M | 6.7M |

Seven times the examples for 1.2x the tokens, which is the whole point of the
tier change.

The unpredicted part: micro-batches dropped to non-finite loss went from **30.5%
to 0.1%**. Phase 5 spent a long investigation on those NaNs and concluded
gradient checkpointing caused them. It was really sequence length; at 2.4k tokens
instead of 16k they simply stop happening. The workaround built then -- a graded
skip tolerance and per-category drop reporting -- was treating a symptom.

## The expert fine-tune was never necessary

Every round from the second onward trained 6.88B parameters, of which 4.83B was an
accidental full fine-tune of the routed experts, because gate_proj/up_proj/down_proj
also match their fused tensors and peft's ParamWrapper leaves the base weight
trainable. This round trained the alternative on identical data: attention and
DeltaNet only, rank raised to 128.

| | trainable | final loss | wall time | peak memory | adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| F, experts, r=32 | 6.88B (7.9%) | 0.134-0.156 | 87 min | 83.9 GB | 26 GB |
| **G, attention only, r=128** | **137M (0.17%)** | 0.145-0.159 | **31 min** | **46.5 GB** | **537 MB** |

Level 1, concepts tier, k=4:

| | pass@1 | pass@4 | solved | fast_1.0 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 20.5% | 31.0% | 31/100 | 5/100 |
| F, experts | 29.8% | 46.0% | 46/100 | 9/100 |
| **G, attention only** | 29.2% | 45.0% | 45/100 | 7/100 |

Equivalent within noise, at a fiftieth of the trainable parameters. Everything the
expert fine-tune cost was avoidable: the memory pressure that forced gradient
checkpointing, the 27.5 GB checkpoints, and phase 6's decision to freeze the
experts for reinforcement learning -- which left only 34M trainable and is the
leading explanation for why that round moved nothing. At rank 128 the cheap
configuration has four times that and matches a full fine-tune, so the next
attempt at RL has a configuration that is both affordable to checkpoint and
demonstrably capable of learning this task.

## What did not work

The repair loop stopped producing. Across 5,020 repair attempts on the new task
set it converted 23, against 100 of 744 in phase 3. Two plausible causes, both
probably true: model-E after three rounds of tuning and RL gives its best answer
first and repeats itself when shown an error, and the newly added operator
families are far enough beyond it that a diagnostic does not help. The
consequence is that the trajectory dataset this phase planned for has almost no
material, and the agentic-SFT idea is unproven here rather than disproven.
