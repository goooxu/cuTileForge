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
OUT_NAME="${OUT_NAME:-grpo_glc}"
LOG="${KEEP_GRPO_LOG:-$WS/runs/${OUT_NAME}.log}"
if [ -n "${KEEP_GRPO_LOCK:-}" ]; then
    LOCK="$KEEP_GRPO_LOCK"
elif [ "$OUT_NAME" = "grpo_glc" ]; then
    LOCK="$WS/runs/.keep_grpo.lock"
else
    LOCK="$WS/runs/.keep_${OUT_NAME}.lock"
fi
PIDFILE="${KEEP_GRPO_PIDFILE:-$WS/runs/.keep_${OUT_NAME}.pid}"
POLL_SEC="${POLL_SEC:-60}"
TOTAL="${TOTAL:-60}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLC}"
HIST="$WS/runs/${OUT_NAME}/history.jsonl"
REMOTE_PIDFILE="$WS/runs/.${OUT_NAME}.remote_pid"

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
n = 0
try:
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        n = max(n, rec["iteration"] + 1)
except Exception:
    n = 0
sys.exit(0 if n >= int(sys.argv[2]) else 1)
PY
}

# True if a remote process cmdline contains $1. Skips the checker itself
# (the needle is on that python's argv). Do not pkill -f.
remote_has_cmd() {
    remote "python3 $(printf %q "$FORGE/rl/proc_has.py") $(printf %q "$1")" >/dev/null 2>&1
}

remote_cmd_pid() {
    remote "python3 $(printf %q "$FORGE/rl/proc_has.py") $(printf %q "$1")" 2>/dev/null
}

job_running() {
    local names rpid args
    names="$(remote "docker ps --format '{{.Names}}'" 2>/dev/null || true)"
    echo "$names" | grep -q '^glc_front_' && return 0
    echo "$names" | grep -qx grpo && return 0
    echo "$names" | grep -qx rlmerge && return 0
    # ps on the GPU box, grep here, so the scanner argv cannot match.
    args="$(remote "ps -eo args=" 2>/dev/null || true)"
    echo "$args" | grep -qE 'run_gl_grpo\.sh|supervise_grpo\.sh|refresh_loop\.sh' && return 0
    if [[ -f "$REMOTE_PIDFILE" ]]; then
        rpid="$(tr -dc '0-9' < "$REMOTE_PIDFILE")"
        if [[ "$rpid" =~ ^[0-9]+$ ]] && remote "test -d /proc/$rpid"; then
            return 0
        fi
    fi
    remote_has_cmd "run_gl_grpo.sh" && return 0
    remote_has_cmd "supervise_grpo.sh" && return 0
    remote_has_cmd "refresh_loop.sh" && return 0
    return 1
}

leftover_vllm() {
    remote "docker ps -a --format '{{.Names}}' | grep -qx qwen-vllm" >/dev/null 2>&1
}

unstick() {
    echo "[keep-grpo] removing leftover vLLM/front/grpo containers on train host"
    remote "docker rm -f qwen-vllm grpo rlmerge glc_front_86 glc_front_87 glc_front_92 glc_front_93 >/dev/null 2>&1 || true"
}

start_grpo() {
    echo "[keep-grpo] starting run_gl_grpo.sh on $TRAIN_HOST out=$OUT_NAME seed=${RL_SEED:-0}"
    # Double-fork on the remote so this SSH returns. `cmd & echo $!` inside
    # bash -c without job control waits for the child, which glued the first
    # watchdog to the whole GRPO run.
    # The remote python starts with a login env, not this watchdog's exports,
    # so OUT_NAME / RL_SEED / MODEL have to go on argv.
    remote "python3 -c $(printf %q "
import os, sys
ws, script, log, total, out_name, rl_seed, model = sys.argv[1:8]
if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)
os.chdir(ws)
env = os.environ.copy()
env['CUTILE_WS'] = ws
env['TOTAL'] = total
env['OUT_NAME'] = out_name
env['RL_SEED'] = rl_seed
env['MODEL'] = model
out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(out, 1)
os.dup2(out, 2)
os.close(out)
devnull = os.open('/dev/null', os.O_RDONLY)
os.dup2(devnull, 0)
os.close(devnull)
os.execve('/bin/bash', ['bash', script], env)
") $(printf %q "$WS") $(printf %q "$FORGE/rl/run_gl_grpo.sh") $(printf %q "$LOG") $(printf %q "$TOTAL") $(printf %q "$OUT_NAME") $(printf %q "${RL_SEED:-0}") $(printf %q "$MODEL")"
    local rpid="" i
    # run_gl_grpo writes the pidfile after flock, then execs into supervise.
    # Wait for that file (NFS) rather than scanning for a script name that
    # disappears. Do not overwrite a live pidfile with a failed scan.
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        if [[ -f "$REMOTE_PIDFILE" ]]; then
            rpid="$(tr -dc '0-9' < "$REMOTE_PIDFILE")"
            if [[ "$rpid" =~ ^[0-9]+$ ]] && remote "test -d /proc/$rpid"; then
                echo "[keep-grpo] remote pid $rpid"
                return
            fi
        fi
        rpid="$(remote_cmd_pid "supervise_grpo.sh" || true)"
        [[ "$rpid" =~ ^[0-9]+$ ]] && break
        rpid="$(remote_cmd_pid "run_gl_grpo.sh" || true)"
        [[ "$rpid" =~ ^[0-9]+$ ]] && break
        sleep 1
    done
    echo "[keep-grpo] remote pid ${rpid:-unknown}"
    [[ "$rpid" =~ ^[0-9]+$ ]] && echo "$rpid" > "$REMOTE_PIDFILE"
}

echo "[keep-grpo] local watchdog host=$TRAIN_HOST total=$TOTAL"
STARTED_AT=0
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
    now="$(date +%s)"
    if leftover_vllm && [ "$STARTED_AT" -gt 0 ] && [ $((now - STARTED_AT)) -lt 180 ]; then
        echo "[keep-grpo] vLLM present, $((now - STARTED_AT))s grace after start"
        sleep "$POLL_SEC"
        continue
    fi
    if leftover_vllm; then
        # Parent may still be in the unit tests / writing the pidfile.
        sleep 15
        if job_running; then
            continue
        fi
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
    STARTED_AT="$(date +%s)"
    sleep 20
done
