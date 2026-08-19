#!/usr/bin/env bash
# Keep a table-A eval running until the verified jsonl exists.
#
# compare_eval_suite.sh is resumable (kernels already on disk; a finished tag
# is skipped) but nothing relaunches it after the parent bash dies. That is
# how a finished train sat on idle GPUs for hours. This loop is the host-side
# insurance: if the eval is not running and not done, start it.
#
# Stuck generate containers are docker rm -f of *named* containers only.
# Do not pkill -f: that pattern matches the ssh command line that launched
# the job and kills the wrong process.
#
# Usage:
#   CUTILE_WS=... rl/keep_eval_alive.sh GLC:/path/to/merged-model
#   KEEP_EVAL_LOG=runs/eval_glc.log STALL_SEC=1500 ... keep_eval_alive.sh ...
set -uo pipefail

SPEC="${1:?usage: keep_eval_alive.sh TAG:PATH}"
TAG="${SPEC%%:*}"
MODEL="${SPEC#*:}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LOG="${KEEP_EVAL_LOG:-$WS/runs/eval_${TAG}.log}"
LOCK="$WS/runs/.keep_eval_${TAG}.lock"
PIDFILE="$WS/runs/.keep_eval_${TAG}.pid"
STALL_SEC="${STALL_SEC:-1500}"
POLL_SEC="${POLL_SEC:-60}"
EXPECTED="${EXPECTED_KERNELS:-3636}"

verified="$WS/runs/${TAG}_l60_verified.jsonl"
kdir="$WS/runs/${TAG}_l60"
level60="$FORGE/kernelbench/KernelBench/level60"

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep] another keep_eval_alive for $TAG holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

done_eval() {
    [[ -f "$verified" ]] || return 1
    local n
    n=$(wc -l < "$verified")
    [[ "$n" -ge "$EXPECTED" ]]
}

n_kernels() {
    python3 - "$kdir" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(1 for f in os.listdir(d) if f.endswith("_kernel.py")) if os.path.isdir(d) else 0)
PY
}

n_problems() {
    python3 - "$level60" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(1 for f in os.listdir(d) if f.endswith(".py")) if os.path.isdir(d) else 0)
PY
}

docker_up() {
    docker ps --format '{{.Names}}' | grep -qx "$1"
}

# True if the eval pipeline (ours or a sibling launched earlier) is alive.
eval_running() {
    pgrep -f "compare_eval_suite.sh ${TAG}:" >/dev/null 2>&1 && return 0
    pgrep -f "run_eval_suite.sh ${TAG}$" >/dev/null 2>&1 && return 0
    pgrep -f "run_eval_suite.sh ${TAG} " >/dev/null 2>&1 && return 0
    docker_up "gen-${TAG}_l60" && return 0
    docker_up "fv_${TAG}_l60" && return 0
    return 1
}

newest_kernel_age() {
    python3 - "$kdir" <<'PY'
import os, sys, time
d = sys.argv[1]
if not os.path.isdir(d):
    print(10 ** 9)
    raise SystemExit
mt = [os.path.getmtime(os.path.join(d, f))
      for f in os.listdir(d) if f.endswith("_kernel.py")]
print(int(time.time() - max(mt)) if mt else 10 ** 9)
PY
}

unstick() {
    echo "[keep] removing gen/fv/vllm containers for $TAG"
    docker rm -f "gen-${TAG}_l60" "fv_${TAG}_l60" qwen-vllm >/dev/null 2>&1 || true
}

start_eval() {
    local skip=0
    if [[ "$(n_problems)" -eq 909 ]]; then
        skip=1
    fi
    echo "[keep] starting compare_eval_suite.sh $SPEC skip_install=$skip"
    SKIP_INSTALL="$skip" CUTILE_WS="$WS" \
        bash "$FORGE/rl/compare_eval_suite.sh" "$SPEC" >>"$LOG" 2>&1 || true
}

echo "[keep] watching $TAG model=$MODEL log=$LOG stall=${STALL_SEC}s"
while true; do
    if done_eval; then
        echo "[keep] complete: $verified ($(wc -l < "$verified") lines)"
        exit 0
    fi
    if eval_running; then
        got="$(n_kernels)"
        age="$(newest_kernel_age)"
        if docker_up "gen-${TAG}_l60" && [[ "$got" -lt "$EXPECTED" ]] \
                && [[ "$age" -ge "$STALL_SEC" ]]; then
            echo "[keep] generate stalled: $got kernels, newest ${age}s ago"
            unstick
        fi
        sleep "$POLL_SEC"
        continue
    fi
    if docker_up qwen-vllm || docker_up "gen-${TAG}_l60" || docker_up "fv_${TAG}_l60"; then
        echo "[keep] leftover containers, no parent; cleaning"
        unstick
    fi
    if [[ ! -d "$MODEL" ]]; then
        echo "[keep] ERROR: model missing: $MODEL"
        sleep "$POLL_SEC"
        continue
    fi
    start_eval
    sleep 15
done
