#!/usr/bin/env bash
# Workspace-host supervisor for GL-G ORPO then table A.
# Starts SFT/ORPO on the train box if model-GLG is missing, then table A
# on the eval box. Compare against GL-F (and GL-E). Do not pkill -f.
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LOCK="$WS/runs/.keep_glg_pipeline.lock"
PIDFILE="$WS/runs/.keep_glg_pipeline.pid"
POLL_SEC="${POLL_SEC:-300}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLG}"
BASE="${BASE:-/raid/tmp/gemsg-cutile/model-GLF}"

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
    echo "[keep-glg] another pipeline watchdog holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote_train() { ssh "${ssh_opts[@]}" "$TRAIN_HOST" "$@"; }

done_eval() {
    [[ -f "$WS/runs/GLG_l60_verified.jsonl" ]] || return 1
    grep -q '^verify done:' "$WS/runs/eval_GLG.log" 2>/dev/null || return 1
    python3 - "$WS/runs/GLG_l60_verified.jsonl" "$FORGE/verify" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
sys.exit(0 if timing_complete(sys.argv[1], need=3636) else 1)
PY
}

echo "[keep-glg] waiting for ORPO merge, then table A"
while true; do
    if done_eval; then
        echo "[keep-glg] table A complete"
        exit 0
    fi
    if ! remote_train "test -f $(printf %q "$MODEL/processor_config.json")"; then
        echo "[keep-glg] starting ORPO on train box"
        remote_train "cd $(printf %q "$WS") && setsid env CUTILE_WS=$(printf %q "$WS") \
            MODEL=$(printf %q "$BASE") \
            bash $(printf %q "$FORGE/rl/run_glg_pipeline.sh") \
            >> $(printf %q "$WS/runs/glg_pipeline.log") 2>&1 < /dev/null & echo \$!"
        sleep "$POLL_SEC"
        continue
    fi
    if [[ -f "$WS/runs/eval_lent" ]]; then
        echo "[keep-glg] eval box lent out; not starting table A"
        sleep "$POLL_SEC"
        continue
    fi
    if [[ ! -f "$WS/runs/.keep_eval_GLG.pid" ]] || ! kill -0 "$(cat "$WS/runs/.keep_eval_GLG.pid")" 2>/dev/null; then
        echo "[keep-glg] starting table A"
        setsid env -u MERGE_ADAPTER -u MERGE_BASE \
            CUTILE_WS="$WS" SKIP_INSTALL=1 \
            MERGE_BASE="$BASE" MERGE_ADAPTER="$WS/models/lora-GLG" \
            bash "$FORGE/rl/keep_eval_alive.sh" "GLG:$MODEL" \
            >> "$WS/runs/keep_eval_GLG.log" 2>&1 < /dev/null &
        echo $! > "$WS/runs/.keep_eval_GLG.launch.pid"
    fi
    sleep "$POLL_SEC"
done
