#!/usr/bin/env bash
# Keep a table-A eval alive across GPU-machine reboots.
#
# Run this on the workspace host (the machine that outlives the GPU box),
# not on the GPU box itself. Kernel files and the verified jsonl live on
# NFS; the GPU box only holds docker, vLLM, and the merged weights on local
# NVMe. This loop SSHs to EVAL_HOST and relaunches merge/eval when that box
# comes back empty. A watchdog that itself runs on the GPU box dies in the
# same reboot it is supposed to recover from.
#
# EVAL_HOST is an ssh target. It is read from the environment or from
# $CUTILE_WS/runs/eval_host (that file is under /runs, which is not
# committed). Do not hard-code machine addresses in this script.
#
# Stuck generate containers are docker rm -f of *named* containers only.
# Do not pkill -f: that pattern matches the ssh command line that launched
# the job and kills the wrong process.
#
# Usage:
#   CUTILE_WS=... EVAL_HOST=gpu-box rl/keep_eval_alive.sh GLC:/raid/.../model-GLC
#   MERGE_BASE=/raid/.../model-GLB MERGE_ADAPTER=/ws/models/lora-GLC \
#     keep_eval_alive.sh GLC:/raid/.../model-GLC
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
MERGE_BASE="${MERGE_BASE:-}"
MERGE_ADAPTER="${MERGE_ADAPTER:-}"

if [[ -z "${EVAL_HOST:-}" && -f "$WS/runs/eval_host" ]]; then
    EVAL_HOST="$(sed -n '/[^[:space:]]/ {s/[[:space:]]*$//; p; q;}' "$WS/runs/eval_host")"
fi
if [[ -z "${EVAL_HOST:-}" ]]; then
    echo "error: set EVAL_HOST or write the ssh target to $WS/runs/eval_host" >&2
    exit 1
fi

verified="$WS/runs/${TAG}_l60_verified.jsonl"
kdir="$WS/runs/${TAG}_l60"
level60="$FORGE/kernelbench/KernelBench/level60"
ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep] another keep_eval_alive for $TAG holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote() {
    ssh "${ssh_opts[@]}" "$EVAL_HOST" "$@"
}

host_up() {
    remote true >/dev/null 2>&1
}

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

# True if a remote process cmdline contains $1. Reads /proc directly:
# pgrep -f is a regex, and leftover detection must not miss the parent
# during the generate→verify handoff (gen gone, fv not started yet).
remote_has_cmd() {
    remote "python3 -c $(printf %q "import os,sys
needle=sys.argv[1]
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd=open('/proc/%s/cmdline'%pid,'rb').read().replace(b'\\0',b' ').decode()
    except OSError:
        continue
    if needle in cmd:
        sys.exit(0)
sys.exit(1)") $(printf %q "$1")" >/dev/null 2>&1
}

# Prints running / leftover / idle / down.
remote_state() {
    if ! host_up; then
        echo down
        return 0
    fi
    local names pidfile rpid
    names="$(remote "docker ps --format '{{.Names}}'" 2>/dev/null || true)"
    has() { echo "$names" | grep -qx "$1"; }
    if has "gen-${TAG}_l60" || has "fv_${TAG}_l60"; then
        echo running
        return 0
    fi
    pidfile="$WS/runs/.eval_${TAG}.remote_pid"
    if [[ -f "$pidfile" ]]; then
        rpid="$(cat "$pidfile")"
        if [[ "$rpid" =~ ^[0-9]+$ ]] && remote "test -d /proc/$rpid"; then
            echo running
            return 0
        fi
    fi
    if remote_has_cmd "compare_eval_suite.sh ${TAG}:" \
            || remote_has_cmd "run_eval_suite.sh ${TAG}"; then
        echo running
        return 0
    fi
    if has qwen-vllm; then
        echo leftover
        return 0
    fi
    echo idle
}

docker_up() {
    remote "docker ps --format '{{.Names}}' | grep -qx $(printf %q "$1")" >/dev/null 2>&1
}

unstick() {
    echo "[keep] removing gen/fv/vllm containers on $EVAL_HOST"
    remote "docker rm -f gen-${TAG}_l60 fv_${TAG}_l60 qwen-vllm >/dev/null 2>&1 || true"
}

as_container_path() {
    local p="$1"
    if [[ "$p" == "$WS"/* ]]; then
        echo "/ws/${p#"$WS"/}"
    else
        echo "$p"
    fi
}

ensure_model() {
    if remote "test -d $(printf %q "$MODEL") && test -f $(printf %q "$MODEL/processor_config.json")"; then
        return 0
    fi
    if [[ -z "$MERGE_BASE" || -z "$MERGE_ADAPTER" ]]; then
        echo "[keep] model missing on $EVAL_HOST ($MODEL) and MERGE_BASE/ADAPTER unset"
        return 1
    fi
    if ! remote "test -d $(printf %q "$MERGE_BASE")"; then
        echo "[keep] merge base missing on $EVAL_HOST: $MERGE_BASE"
        return 1
    fi
    local adapter
    adapter="$(as_container_path "$MERGE_ADAPTER")"
    echo "[keep] re-merging $adapter onto $MERGE_BASE -> $MODEL"
    remote "cd $(printf %q "$WS") && IMAGE=cutile-train:latest MOUNTS='-v /raid/tmp:/raid/tmp' GPUS=none \
        $(printf %q "$FORGE/kernelbench/scripts/in_container.sh") \
        $(printf %q "cd /ws/cuTileForge && python3 -u train/merge_lora.py --base $MERGE_BASE --adapter $adapter --out $MODEL")"
    remote "test -f $(printf %q "$MODEL/processor_config.json")"
}

start_eval() {
    local skip=0
    if [[ "$(n_problems)" -eq 909 ]]; then
        skip=1
    fi
    echo "[keep] starting compare_eval_suite.sh $SPEC on $EVAL_HOST skip_install=$skip"
    local rpid
    rpid="$(remote "cd $(printf %q "$WS") && setsid nohup env SKIP_INSTALL=$skip CUTILE_WS=$(printf %q "$WS") \
        bash $(printf %q "$FORGE/rl/compare_eval_suite.sh") $(printf %q "$SPEC") \
        >> $(printf %q "$LOG") 2>&1 < /dev/null & echo \$!")"
    echo "[keep] remote pid $rpid"
    [[ "$rpid" =~ ^[0-9]+$ ]] && echo "$rpid" > "$WS/runs/.eval_${TAG}.remote_pid"
}

echo "[keep] local watchdog TAG=$TAG host=$EVAL_HOST model=$MODEL stall=${STALL_SEC}s"
while true; do
    if done_eval; then
        echo "[keep] complete: $verified ($(wc -l < "$verified") lines)"
        exit 0
    fi
    if ! host_up; then
        echo "[keep] $EVAL_HOST unreachable; retry in ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    state="$(remote_state)"
    case "$state" in
        running)
            got="$(n_kernels)"
            age="$(newest_kernel_age)"
            if docker_up "gen-${TAG}_l60" && [[ "$got" -lt "$EXPECTED" ]] \
                    && [[ "$age" -ge "$STALL_SEC" ]]; then
                echo "[keep] generate stalled: $got kernels, newest ${age}s ago"
                unstick
            fi
            sleep "$POLL_SEC"
            ;;
        leftover)
            echo "[keep] leftover containers, no parent"
            unstick
            sleep 5
            ;;
        idle)
            if ! ensure_model; then
                sleep "$POLL_SEC"
                continue
            fi
            start_eval
            sleep 15
            ;;
        down)
            echo "[keep] $EVAL_HOST went down mid-check; retry in ${POLL_SEC}s"
            sleep "$POLL_SEC"
            ;;
        *)
            echo "[keep] unexpected state: $state"
            sleep "$POLL_SEC"
            ;;
    esac
done
