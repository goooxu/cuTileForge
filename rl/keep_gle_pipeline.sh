#!/usr/bin/env bash
# Workspace-host supervisor for the GL-E mix/SFT/eval tail.
# Harvests have their own keep_harvest_alive. This waits for those jsonl,
# refuses a failed speed gate, then SFT on the train box and table A on the
# eval box.
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LOG="${KEEP_GLE_LOG:-$WS/runs/keep_gle_pipeline.log}"
LOCK="$WS/runs/.keep_gle_pipeline.lock"
PIDFILE="$WS/runs/.keep_gle_pipeline.pid"
POLL_SEC="${POLL_SEC:-300}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLE}"
BASE="${BASE:-/raid/tmp/gemsg-cutile/model-GLC}"

if [[ -z "${TRAIN_HOST:-}" && -f "$WS/runs/train_host" ]]; then
    TRAIN_HOST="$(sed -n '/[^[:space:]]/ {s/[[:space:]]*$//; p; q;}' "$WS/runs/train_host")"
fi
if [[ -z "${EVAL_HOST:-}" && -f "$WS/runs/eval_host" ]]; then
    EVAL_HOST="$(sed -n '/[^[:space:]]/ {s/[[:space:]]*$//; p; q;}' "$WS/runs/eval_host")"
fi
ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep-gle] another pipeline watchdog holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote_train() { ssh "${ssh_opts[@]}" "$TRAIN_HOST" "$@"; }
remote_eval() { ssh "${ssh_opts[@]}" "$EVAL_HOST" "$@"; }

done_eval() {
    [[ -f "$WS/runs/GLE_l60_verified.jsonl" ]] || return 1
    grep -q '^verify done:' "$WS/runs/eval_GLE.log" 2>/dev/null || return 1
    python3 - "$WS/runs/GLE_l60_verified.jsonl" <<'PY'
import json, sys
n = t = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("passed"):
        n += 1
        if r.get("speedup") is not None:
            t += 1
sys.exit(0 if n and t >= n * 0.99 else 1)
PY
}

files_ready() {
    local f
    for f in harvest_glc86 harvest_glc87 harvest_glc92 harvest_glc93 harvest_glc80; do
        [[ -f "$WS/runs/${f}_verified.jsonl" ]] || return 1
        [[ "$(wc -l < "$WS/runs/${f}_verified.jsonl")" -gt 0 ]] || return 1
    done
    python3 - "$WS/runs/harvest_glc80_verified.jsonl" <<'PY'
import json, sys
n = t = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("passed"):
        n += 1
        if r.get("speedup") is not None:
            t += 1
sys.exit(0 if n and t >= n * 0.99 else 1)
PY
}

echo "[keep-gle] waiting for harvests, then SFT/eval"
while true; do
    if done_eval; then
        echo "[keep-gle] table A complete"
        exit 0
    fi
    if ! files_ready; then
        echo "[keep-gle] harvests not ready; sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    if ! python3 "$FORGE/rl/speed_gate.py" \
            --verified "$WS/runs/harvest_glc80_verified.jsonl" --level 80 \
            | tee -a "$WS/runs/gle_speed_gate.txt"; then
        echo "[keep-gle] speed gate failed; stopping"
        exit 1
    fi
    if ! remote_train "test -f $(printf %q "$MODEL/processor_config.json")"; then
        echo "[keep-gle] starting SFT on train box"
        remote_train "cd $(printf %q "$WS") && setsid env CUTILE_WS=$(printf %q "$WS") \
            MODEL=$(printf %q "$BASE") \
            bash $(printf %q "$FORGE/rl/run_gle_pipeline.sh") \
            >> $(printf %q "$WS/runs/gle_pipeline.log") 2>&1 < /dev/null & echo \$!"
        sleep "$POLL_SEC"
        continue
    fi
    if [[ ! -f "$WS/runs/.keep_eval_GLE.pid" ]] || ! kill -0 "$(cat "$WS/runs/.keep_eval_GLE.pid")" 2>/dev/null; then
        echo "[keep-gle] starting table A"
        # Do not inherit MERGE_ADAPTER from a GRPO shell. GLE remake on the
        # eval box must use the SFT adapter, not grpo_glc_s1.
        setsid env CUTILE_WS="$WS" SKIP_INSTALL=1 \
            MERGE_BASE="$BASE" MERGE_ADAPTER="$WS/models/lora-GLE" \
            bash "$FORGE/rl/keep_eval_alive.sh" "GLE:$MODEL" \
            >> "$WS/runs/keep_eval_GLE.log" 2>&1 < /dev/null &
        echo $! > "$WS/runs/.keep_eval_GLE.launch.pid"
    fi
    sleep "$POLL_SEC"
done
