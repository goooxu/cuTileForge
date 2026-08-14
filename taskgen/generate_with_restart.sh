#!/usr/bin/env bash
# Sample from the model, restarting the server whenever its engine dies.
#
# The vLLM build that works on GB200 still hits "CUDA error: an illegal memory
# access" under sustained load and takes the engine down with it. Sampling is
# resumable -- generate_samples.py skips problems whose kernel file already
# exists -- so the fix is to relaunch and continue rather than lose the run.
#
# Usage: ./generate_with_restart.sh <run_name> <level> <num_samples> [max_rounds]
set -uo pipefail

RUN_NAME="${1:?usage: generate_with_restart.sh <run_name> <level> <num_samples> [max_rounds]}"
LEVEL="${2:?}"
NUM_SAMPLES="${3:?}"
MAX_ROUNDS="${4:-12}"

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:-$(dirname "$FORGE")}"
KB="$FORGE/kernelbench"
RUN_DIR="$WS/runs/$RUN_NAME"

export CUTILE_WS="$WS"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"

expected() {
    python3 - "$LEVEL" "$NUM_SAMPLES" "$KB" <<'PY'
import os, sys
level, n, kb = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
d = os.path.join(kb, "KernelBench", "level%d" % level)
print(len([f for f in os.listdir(d) if f.endswith(".py")]) * n)
PY
}

have() {
    python3 - "$RUN_DIR" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(1 for f in os.listdir(d) if f.endswith("_kernel.py")) if os.path.isdir(d) else 0)
PY
}

GEN_CONTAINER="gen-$RUN_NAME"

server_up() { curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; }

# Sampling runs inside a container, so killing this script from the host leaves
# the client alive and still hitting the server. Name it so it can be reaped.
cleanup() {
    docker rm -f "$GEN_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
cleanup

start_server() {
    echo "[restart] (re)starting vLLM"
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
    "$KB/scripts/serve_qwen.sh" >/dev/null 2>&1
    for _ in $(seq 1 180); do
        server_up && { echo "[restart] server up"; return 0; }
        docker ps --filter name=qwen-vllm --format '{{.Names}}' | grep -q qwen-vllm || {
            echo "[restart] container died during startup"; return 1; }
        sleep 10
    done
    echo "[restart] server did not come up in time"
    return 1
}

TARGET="$(expected)"
echo "target: $TARGET samples for $RUN_NAME (level $LEVEL x $NUM_SAMPLES)"

for round in $(seq 1 "$MAX_ROUNDS"); do
    got="$(have)"
    echo "=== round $round: have $got / $TARGET ==="
    [[ "$got" -ge "$TARGET" ]] && { echo "complete"; exit 0; }

    server_up || start_server || { echo "cannot start server, giving up"; exit 1; }

    # generate_samples.py exits non-zero on engine death; that is expected here.
    NAME="$GEN_CONTAINER" "$KB/scripts/run_generate.sh" "$RUN_NAME" "$LEVEL" "$NUM_SAMPLES" \
        log_raw_response=True 2>&1 | tail -3
    cleanup

    after="$(have)"
    if [[ "$after" -le "$got" ]]; then
        echo "[restart] no progress this round; restarting server"
        docker rm -f qwen-vllm >/dev/null 2>&1 || true
    fi
done

echo "stopped after $MAX_ROUNDS rounds with $(have) / $TARGET samples"
