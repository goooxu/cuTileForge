#!/usr/bin/env bash
# Run a command inside the cuTile eval container on the dev machine.
#
# The workspace is typically an NFS mount with root squashed, so the container
# must run as the invoking user or it cannot even write a file.
#
# Set DETACH=1 for long jobs. An attached `docker run` dies with its ssh session,
# which is how a multi-hour eval was lost when a time-limited dev machine's
# allocation expired mid-run.
#
# Usage: ./in_container.sh <command...>
#        GPUS=none ./in_container.sh python3 scripts/foo.py
#        DETACH=1 NAME=eval ./in_container.sh python3 scripts/bar.py
set -euo pipefail

# Workspace root: the directory holding models/ and runs/, mounted into the
# container. Defaults to the parent of the KernelBench checkout, which is the
# cutile-forge repo when set up via scripts/setup_kernelbench.sh.
WS="${CUTILE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPUS="${GPUS:-all}"
NAME="${NAME:-}"
DETACH="${DETACH:-0}"

args=(--user "$(id -u):$(id -g)" --ipc=host --network host
      -e HOME=/tmp
      -e PYTHONPATH=/ws/cutile-eval/src
      -e HF_HOME=/ws/models/hf-cache
      -e NVIDIA_TF32_OVERRIDE=0
      # The OpenAI client refuses to construct without a key; the local vLLM
      # server is started without --api-key and ignores its value.
      -e SGLANG_API_KEY=local-no-auth
      -v "$WS":/ws
      -w /ws/cutile-eval)

# GPUS=none skips GPU passthrough entirely, for CPU-only steps such as generation
# (which just talks to the vLLM server over HTTP).
[[ "$GPUS" != "none" ]] && args+=(--gpus "$GPUS")
[[ -n "$NAME" ]] && args+=(--name "$NAME")

# Detached runs keep their logs (docker logs) instead of being auto-removed.
if [[ "$DETACH" == "1" ]]; then
    args+=(-d)
else
    args+=(--rm)
fi

# A single argument is treated as a shell command string (so pipelines and &&
# work). Multiple arguments are treated as argv and escaped individually, which
# is what keeps quoting intact for values like gpu_arch=['Blackwell'].
if [[ $# -eq 1 ]]; then
    cmd="$1"
else
    cmd="$(printf '%q ' "$@")"
fi

exec docker run "${args[@]}" --entrypoint bash cutile-eval:latest -lc "$cmd"
