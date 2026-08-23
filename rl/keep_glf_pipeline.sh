#!/usr/bin/env bash
# Workspace-host supervisor for the GL-F speed SFT/eval tail.
# Harvests have their own keep_harvest_alive. This waits for those jsonl,
# then SFT on the train box (slow-vs-compile + kernel_ms-spread slice) and
# table A on the eval box.
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LOCK="$WS/runs/.keep_glf_pipeline.lock"
PIDFILE="$WS/runs/.keep_glf_pipeline.pid"
POLL_SEC="${POLL_SEC:-300}"
MODEL="${MODEL:-/raid/tmp/gemsg-cutile/model-GLF}"
BASE="${BASE:-/raid/tmp/gemsg-cutile/model-GLE}"

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
    echo "[keep-glf] another pipeline watchdog holds $LOCK; exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote_train() { ssh "${ssh_opts[@]}" "$TRAIN_HOST" "$@"; }

done_eval() {
    [[ -f "$WS/runs/GLF_l60_verified.jsonl" ]] || return 1
    grep -q '^verify done:' "$WS/runs/eval_GLF.log" 2>/dev/null || return 1
    python3 - "$WS/runs/GLF_l60_verified.jsonl" <<'PY'
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
    for f in harvest_gle86 harvest_gle87 harvest_gle92 harvest_gle93; do
        [[ -f "$WS/runs/${f}_verified.jsonl" ]] || return 1
        [[ "$(wc -l < "$WS/runs/${f}_verified.jsonl")" -gt 0 ]] || return 1
    done
    python3 - "$FORGE/verify" \
        "$WS/runs/harvest_gle86_verified.jsonl" \
        "$WS/runs/harvest_gle87_verified.jsonl" \
        "$WS/runs/harvest_gle92_verified.jsonl" \
        "$WS/runs/harvest_gle93_verified.jsonl" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from fast_verify import timing_complete
for path in sys.argv[2:]:
    if not timing_complete(path):
        raise SystemExit(1)
PY
}

echo "[keep-glf] waiting for timed harvests, then SFT/eval"
while true; do
    if done_eval; then
        echo "[keep-glf] table A complete"
        exit 0
    fi
    if ! files_ready; then
        echo "[keep-glf] harvests not ready; sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi
    if ! remote_train "test -f $(printf %q "$MODEL/processor_config.json")"; then
        echo "[keep-glf] starting SFT on train box"
        remote_train "cd $(printf %q "$WS") && setsid env CUTILE_WS=$(printf %q "$WS") \
            MODEL=$(printf %q "$BASE") \
            bash $(printf %q "$FORGE/rl/run_glf_pipeline.sh") \
            >> $(printf %q "$WS/runs/glf_pipeline.log") 2>&1 < /dev/null & echo \$!"
        sleep "$POLL_SEC"
        continue
    fi
    if [[ ! -f "$WS/runs/.keep_eval_GLF.pid" ]] || ! kill -0 "$(cat "$WS/runs/.keep_eval_GLF.pid")" 2>/dev/null; then
        echo "[keep-glf] starting table A"
        setsid env -u MERGE_ADAPTER -u MERGE_BASE \
            CUTILE_WS="$WS" SKIP_INSTALL=1 \
            MERGE_BASE="$BASE" MERGE_ADAPTER="$WS/models/lora-GLF" \
            bash "$FORGE/rl/keep_eval_alive.sh" "GLF:$MODEL" \
            >> "$WS/runs/keep_eval_GLF.log" 2>&1 < /dev/null &
        echo $! > "$WS/runs/.keep_eval_GLF.launch.pid"
    fi
    sleep "$POLL_SEC"
done
