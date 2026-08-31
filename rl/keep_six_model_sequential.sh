#!/usr/bin/env bash
# Workspace-host watchdog for the resumable six-model sequential screen.
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WS/runs/.keep_six_model_sequential.lock"
PIDFILE="$WS/runs/.keep_six_model_sequential.pid"
REMOTE_PID="$WS/runs/.six_model_sequential.pid"
LOG="$WS/runs/six_model_sequential.log"
POLL_SEC="${POLL_SEC:-180}"

if [[ -z "${TRAIN_HOST:-}" && -f "$WS/runs/train_host" ]]; then
    TRAIN_HOST="$(python3 - "$WS/runs/train_host" <<'PY'
import sys
print(next(line.strip() for line in open(sys.argv[1]) if line.strip()))
PY
)"
fi
ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[keep-sequential] another watchdog is active"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

remote() { ssh "${ssh_opts[@]}" "$TRAIN_HOST" "$@"; }

done_all() {
    [[ -f "$LOG" ]] || return 1
    python3 - "$LOG" <<'PY'
import sys
text = open(sys.argv[1], errors="replace").read()
raise SystemExit(0 if "six-model sequential screen complete" in text else 1)
PY
}

remote_alive() {
    [[ -f "$REMOTE_PID" ]] || return 1
    local pid
    pid="$(<"$REMOTE_PID")"
    remote "test -d /proc/$(printf %q "$pid")"
}

echo "[keep-sequential] supervising six-model screen"
while true; do
    if done_all; then
        echo "[keep-sequential] complete"
        exit 0
    fi
    if ! remote_alive; then
        echo "[keep-sequential] relaunching resumable pipeline"
        remote "cd $(printf %q "$FORGE") && \
            setsid -f env CUTILE_WS=$(printf %q "$WS") \
            bash $(printf %q "$FORGE/rl/run_six_model_sequential.sh") \
            >> $(printf %q "$LOG") 2>&1 < /dev/null"
    fi
    sleep "$POLL_SEC"
done
