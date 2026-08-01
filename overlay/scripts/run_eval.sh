#!/usr/bin/env bash
# Evaluation phase: compile, check and time every generated kernel on GPU.
#
# Requires the vLLM server to be stopped first, since this uses all four GPUs.
# build_cache is off because cuTile JIT-compiles at first launch rather than
# through a separate CPU-side build step.
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
    timeout=300 \
    build_cache=False \
    kernel_eval_build_dir=/tmp/kbcache \
    "$@"
