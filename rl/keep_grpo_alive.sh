#!/usr/bin/env bash
# Keep GL-C GRPO alive across GPU-machine reboots.
#
# Run this on the workspace host, not on the GPU box. Same reason as
# keep_eval_alive.sh: a watchdog on the training node dies in the reboot it
# is supposed to recover from. TRAIN_HOST is an ssh target from the
# environment or $CUTILE_WS/runs/train_host (under /runs, not committed).
#
# Stuck jobs are docker rm -f of named containers only. Do not pkill -f.
#
# Usage:
#   CUTILE_WS=... TRAIN_HOST=gpu-box rl/keep_grpo_alive.sh
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LOG="${KEEP_GRPO_LOG:-$WS/runs/grpo_glc.log}"
LOCK="$WS/runs/.keep_grpo.lock"
PIDFILE="$WS/runs/.keep_grpo.pid"
POLL_SEC="${POLL_SEC:-60}"
TOTAL="${TOTAL:-60}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLC}"
HIST="$WS/runs/grpo_glc/history.jsonl"
REMOTE_PIDFILE="$WS/runs/.grpo_glc.remote_pid"

if [[ -z "${TRAIN_HOST:-}" && -f "$WS/runs/train_host" ]]; then
    TRAIN_HOST="$(sed -n '/[^[:space:]]/ {s/[[:space:]]*$//; p; q;}' "$WS/runs/train_host")"
fi
if [[ -z "${TRAIN_HOST:-}" ]]; then
    echo "error: set TRAIN_HOST or write the ssh target to $WS/runs/train_host" >&2
    exit 1
fi

ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep-grpo] another keep_grpo_alive holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote() {
    ssh "${ssh_opts[@]}" "$TRAIN_HOST" "$@"
}

host_up() {
    remote true >/dev/null 2>&1
}

done_grpo() {
    [ -f "$HIST" ] || return 1
    python3 - "$HIST" "$TOTAL" <<'PY'
import json, sys
try:
    n = max(json.loads(l)["iteration"] for l in open(sys.argv[1]) if l.strip()) + 1
except Exception:
    n = 0
sys.exit(0 if n >= int(sys.argv[2]) else 1)
PY
}

job_running() {
    local names rpid
    names="$(remote "docker ps --format '{{.Names}}'" 2>/dev/null || true)"
    echo "$names" | grep -q '^glc_front_' && return 0
    echo "$names" | grep -qx grpo && return 0
    echo "$names" | grep -qx rlmerge && return 0
    if [[ -f "$REMOTE_PIDFILE" ]]; then
        rpid="$(cat "$REMOTE_PIDFILE")"
        if [[ "$rpid" =~ ^[0-9]+$ ]] && remote "test -d /proc/$rpid"; then
            return 0
        fi
    fi
    return 1
}

leftover_vllm() {
    remote "docker ps --format '{{.Names}}' | grep -qx qwen-vllm" >/dev/null 2>&1
}

unstick() {
    echo "[keep-grpo] removing leftover vLLM/front/grpo containers on train host"
    remote "docker rm -f qwen-vllm grpo rlmerge glc_front_86 glc_front_87 glc_front_92 glc_front_93 >/dev/null 2>&1 || true"
}

start_grpo() {
    echo "[keep-grpo] starting run_gl_grpo.sh on $TRAIN_HOST"
    local rpid
    rpid="$(remote "cd $(printf %q "$WS") && setsid env CUTILE_WS=$(printf %q "$WS") TOTAL=$(printf %q "$TOTAL") \
        bash $(printf %q "$FORGE/rl/run_gl_grpo.sh") \
        >> $(printf %q "$LOG") 2>&1 < /dev/null & echo \$!")"
    echo "[keep-grpo] remote pid $rpid"
    [[ "$rpid" =~ ^[0-9]+$ ]] && echo "$rpid" > "$REMOTE_PIDFILE"
}

echo "[keep-grpo] local watchdog host=$TRAIN_HOST total=$TOTAL"
while true; do
    if done_grpo; then
        echo "[keep-grpo] complete: $HIST"
        exit 0
    fi
    if ! host_up; then
        echo "[keep-grpo] train host unreachable; retry in ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    if job_running; then
        sleep "$POLL_SEC"
        continue
    fi
    if leftover_vllm; then
        echo "[keep-grpo] leftover vLLM, no parent"
        unstick
        sleep 5
    fi
    if ! remote "test -f $(printf %q "$MODEL/processor_config.json") || test -f $(printf %q "$MODEL/config.json")"; then
        echo "[keep-grpo] $MODEL not on train host yet; retry in ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    start_grpo
    sleep 20
done
