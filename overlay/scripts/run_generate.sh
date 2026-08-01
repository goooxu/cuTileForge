#!/usr/bin/env bash
# Generation phase: sample cuTile kernels from the served model.
#
# Runs with check_kernel=False on purpose. The static checker asserts, which
# would silently discard any sample that fails it -- including the torch
# passthroughs we specifically want to count. Every sample is kept and the
# cuTile usage gate is applied later, in scripts/analyze_cutile_run.py.
#
# Usage: ./run_generate.sh <run_name> <level> <num_samples> [extra pydra args...]
set -euo pipefail

RUN_NAME="${1:?usage: run_generate.sh <run_name> <level> <num_samples>}"
LEVEL="${2:?}"
NUM_SAMPLES="${3:?}"
shift 3

cd "$(dirname "$0")/.."

GPUS=none scripts/in_container.sh python3 scripts/generate_samples.py \
    run_name="$RUN_NAME" \
    dataset_src=local \
    level="$LEVEL" \
    num_samples="$NUM_SAMPLES" \
    runs_dir=/ws/runs \
    server_type=local_chat \
    model_name=Qwen3-Coder-Next \
    server_address=localhost \
    server_port=8000 \
    temperature=1.0 \
    top_p=0.95 \
    top_k=40 \
    max_tokens=8192 \
    backend=cutile \
    precision=fp32 \
    prompt_option=one_shot \
    custom_prompt_key=cutile_docs \
    num_workers=32 \
    check_kernel=False \
    "$@"
