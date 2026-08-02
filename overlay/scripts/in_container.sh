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

# The KernelBench checkout this script lives in.
CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Workspace root: the directory holding models/ and runs/, mounted at /ws.
# Defaults to the checkout's parent, which is the repo root when set up via
# scripts/setup_kernelbench.sh.
WS="${CUTILE_WS:-$(dirname "$CHECKOUT")}"


# Paths inside the container are derived from the checkout's position under the
# workspace rather than hardcoded, so the same script works whether CUTILE_WS is
# the immediate parent or a higher ancestor.
case "$CHECKOUT" in
    "$WS"/*) REL="${CHECKOUT#"$WS"/}" ;;
    "$WS")   REL="." ;;
    *) echo "error: CUTILE_WS ($WS) is not an ancestor of $CHECKOUT" >&2; exit 1 ;;
esac

GPUS="${GPUS:-all}"
NAME="${NAME:-}"
DETACH="${DETACH:-0}"

args=(--user "$(id -u):$(id -g)" --ipc=host --network host
      -e HOME=/tmp
      # The container has no /etc/passwd entry for the mounted-in uid, so
      # getpass.getuser() raises when torch resolves its cache directory.
      # It checks $USER before falling back to getpwuid().
      -e USER=cutile
      -e PYTHONPATH="/ws/$REL/src"
      -e HF_HOME=/ws/models/hf-cache
      -e NVIDIA_TF32_OVERRIDE=0
      # The OpenAI client refuses to construct without a key; the local vLLM
      # server is started without --api-key and ignores its value.
      -e SGLANG_API_KEY=local-no-auth
      -v "$WS":/ws
      -w "/ws/$REL")

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
