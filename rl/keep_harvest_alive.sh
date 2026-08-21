#!/usr/bin/env bash
# Keep a multi-level GL harvest alive across GPU-box reboots.
#
# Run on the workspace host, not the GPU box. HARVEST_HOST is an ssh target
# from the environment or $CUTILE_WS/runs/{eval,train}_host.
#
# Usage:
#   CUTILE_WS=... HARVEST_WHICH=eval MODEL=/raid/tmp/gemsg-cutile/model-GLC \
#     rl/keep_harvest_alive.sh harvest_glc:86,87,92,93
#   CUTILE_WS=... HARVEST_WHICH=train MEASURE_TIME=1 MODEL=... \
#     rl/keep_harvest_alive.sh harvest_glc80:80
set -uo pipefail

SPEC="${1:?usage: keep_harvest_alive.sh PREFIX:LEVELS}"
PREFIX="${SPEC%%:*}"
LEVELS="${SPEC#*:}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
TAG="${PREFIX}_${LEVELS//,/_}"
LOG="${KEEP_HARVEST_LOG:-$WS/runs/keep_${TAG}.log}"
LOCK="$WS/runs/.keep_${TAG}.lock"
PIDFILE="$WS/runs/.keep_${TAG}.pid"
REMOTE_PIDFILE="$WS/runs/.${TAG}.remote_pid"
POLL_SEC="${POLL_SEC:-60}"
STALL_SEC="${STALL_SEC:-1500}"
K="${K:-8}"
MEASURE_TIME="${MEASURE_TIME:-0}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLC}"
WHICH="${HARVEST_WHICH:-eval}"

if [[ -z "${HARVEST_HOST:-}" ]]; then
    hostfile="$WS/runs/${WHICH}_host"
    if [[ -f "$hostfile" ]]; then
        HARVEST_HOST="$(sed -n '/[^[:space:]]/ {s/[[:space:]]*$//; p; q;}' "$hostfile")"
    fi
fi
if [[ -z "${HARVEST_HOST:-}" ]]; then
    echo "error: set HARVEST_HOST or HARVEST_WHICH=eval|train" >&2
    exit 1
fi

ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep-harvest] another watchdog holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote() {
    ssh "${ssh_opts[@]}" "$HARVEST_HOST" "$@"
}

host_up() {
    remote true >/dev/null 2>&1
}

remote_has_cmd() {
    remote "python3 $(printf %q "$FORGE/rl/proc_has.py") $(printf %q "$1")" >/dev/null 2>&1
}

done_harvest() {
    local lv run out
    IFS=',' read -r -a lv_arr <<< "$LEVELS"
    for lv in "${lv_arr[@]}"; do
        lv="$(echo "$lv" | tr -d '[:space:]')"
        run="${PREFIX}${lv}"
        out="$WS/runs/${run}_verified.jsonl"
        [[ -f "$out" ]] || return 1
        [[ "$(wc -l < "$out")" -gt 0 ]] || return 1
        if [[ "$MEASURE_TIME" == "1" ]]; then
            python3 - "$out" "$FORGE/verify" <<'PY' || return 1
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
sys.exit(0 if timing_complete(sys.argv[1]) else 1)
PY
        fi
    done
    return 0
}

job_running() {
    local names rpid
    names="$(remote "docker ps --format '{{.Names}}'" 2>/dev/null || true)"
    echo "$names" | grep -qE "^(gen-${PREFIX}|fv_${PREFIX}|tv_${PREFIX})" && return 0
    if [[ -f "$REMOTE_PIDFILE" ]]; then
        rpid="$(tr -dc '0-9' < "$REMOTE_PIDFILE")"
        if [[ "$rpid" =~ ^[0-9]+$ ]] && remote "kill -0 $rpid" >/dev/null 2>&1; then
            return 0
        fi
    fi
    remote_has_cmd "run_gl_harvest_levels.sh ${PREFIX}" && return 0
    remote_has_cmd "run_gl_harvest.sh ${PREFIX}" && return 0
    remote_has_cmd "generate_with_restart.sh ${PREFIX}" && return 0
    return 1
}

leftover_vllm() {
    remote "docker ps -a --format '{{.Names}}' | grep -qx qwen-vllm" >/dev/null 2>&1
}

unstick() {
    # Named containers only. Do not pkill -f. Do not rm qwen-vllm out from
    # under a live parent; start_server retries three times.
    echo "[keep-harvest] leftover containers left for the next harvest to reap"
}

newest_kernel_age() {
    python3 - "$WS/runs" "$PREFIX" <<'PY'
import os, sys, time
root, prefix = sys.argv[1], sys.argv[2]
mt = []
for name in os.listdir(root):
    if not name.startswith(prefix):
        continue
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.endswith("_kernel.py"):
            mt.append(os.path.getmtime(os.path.join(d, f)))
print(int(time.time() - max(mt)) if mt else 10 ** 9)
PY
}

start_harvest() {
    echo "[keep-harvest] starting run_gl_harvest_levels.sh $PREFIX $LEVELS on host"
    local rpid
    rpid="$(remote "cd $(printf %q "$WS") && setsid env \
        CUTILE_WS=$(printf %q "$WS") \
        MODEL=$(printf %q "$MODEL") \
        MEASURE_TIME=$(printf %q "$MEASURE_TIME") \
        VLLM_IMAGE=vllm/vllm-openai:muse-glimmer \
        bash $(printf %q "$FORGE/rl/run_gl_harvest_levels.sh") \
        $(printf %q "$PREFIX") $(printf %q "$LEVELS") $(printf %q "$K") \
        >> $(printf %q "$WS/runs/${TAG}.log") 2>&1 < /dev/null & echo \$!")"
    echo "[keep-harvest] remote pid $rpid"
    [[ "$rpid" =~ ^[0-9]+$ ]] && echo "$rpid" > "$REMOTE_PIDFILE"
}

echo "[keep-harvest] prefix=$PREFIX levels=$LEVELS host=$HARVEST_HOST model=$MODEL time=$MEASURE_TIME"
while true; do
    if done_harvest; then
        echo "[keep-harvest] complete: $PREFIX $LEVELS"
        exit 0
    fi
    if ! host_up; then
        echo "[keep-harvest] host unreachable; retry in ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    if job_running; then
        age="$(newest_kernel_age)"
        echo "[keep-harvest] running (newest kernel ${age}s ago)"
        sleep "$POLL_SEC"
        continue
    fi
    if leftover_vllm; then
        echo "[keep-harvest] leftover vLLM, no harvest parent"
        unstick
    fi
    if ! remote "test -f $(printf %q "$MODEL/processor_config.json")"; then
        echo "[keep-harvest] model missing: $MODEL"
        sleep "$POLL_SEC"
        continue
    fi
    start_harvest
    sleep 15
done
