#!/usr/bin/env bash
# Evaluation phase: compile, check and time every generated kernel on GPU.
#
# Requires the vLLM server to be stopped first, since this uses all four GPUs.
# build_cache is off because cuTile JIT-compiles at first launch rather than
# through a separate CPU-side build step.
#
# EVAL_TIMEOUT is per sample. The default was 300, which turned out to dominate
# the wall clock of any large run: one kernel that hangs blocks its worker for
# five minutes, and a Level 1 pass once managed four samples in thirty minutes.
# At 60 the same pass did 428 in twenty-five. Baselines here are sub-millisecond,
# so anything still running after a minute is a failure whichever bound is used.
#
# Usage: ./run_eval.sh <run_name> <level> <num_samples> [extra pydra args...]
set -euo pipefail

RUN_NAME="${1:?usage: run_eval.sh <run_name> <level> <num_samples>}"
LEVEL="${2:?}"
NUM_SAMPLES="${3:?}"
shift 3

cd "$(dirname "$0")/.."

GPUS=all scripts/in_container.sh python3 scripts/eval_from_generations.py \
    run_name="$RUN_NAME" \
    dataset_src=local \
    level="$LEVEL" \
    runs_dir=/ws/runs \
    num_samples_per_problem="$NUM_SAMPLES" \
    backend=cutile \
    precision=fp32 \
    "gpu_arch=['Blackwell']" \
    num_gpu_devices=4 \
    timeout="${EVAL_TIMEOUT:-60}" \
    build_cache=False \
    kernel_eval_build_dir=/tmp/kbcache \
    "$@"
