#!/usr/bin/env bash
# Ensure the six requested checkpoints exist on local NVMe.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
RAID="${RAID_ROOT:-/raid/tmp/gemsg-cutile}"
BASE="$RAID/Qwen3-Coder-Next"
Q="$RAID/model-Q"
Q38="$RAID/Qwen3.8-27B"
LOCK="$WS/runs/.prepare_sequential_models.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another sequential model preparation holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.prepare_sequential_models.pid"

mkdir -p "$RAID"

if [[ ! -f "$BASE/config.json" ]]; then
    if [[ ! -f "$WS/models/Qwen3-Coder-Next/config.json" ]]; then
        echo "workspace base model is missing" >&2
        exit 1
    fi
    echo "=== copy Next base from shared workspace to local NVMe ==="
    rm -rf "$BASE"
    cp -a "$WS/models/Qwen3-Coder-Next" "$BASE"
fi

merge_model() {
    local base="$1"
    local adapter="$2"
    local out="$3"
    local name="$4"
    if [[ -f "$out/config.json" ]]; then
        echo "$name already exists: $out"
        return
    fi
    rm -rf "$out"
    echo "=== merge $name ==="
    CUTILE_WS="$WS" IMAGE=cutile-train:latest GPUS=none \
        MOUNTS="-v /raid/tmp:/raid/tmp" NAME="prepare_${name}" \
        "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u train/merge_lora.py \
            --base $base --adapter $adapter --out $out"
}

if [[ ! -f "$Q/config.json" ]]; then
    F="$RAID/model-F"
    H="$RAID/model-H"
    L="$RAID/model-L"
    M="$RAID/model-M"
    merge_model "$BASE" /ws/models/lora-F-experts "$F" next_F
    merge_model "$F" /ws/models/lora-H-sharpen "$H" next_H
    rm -rf "$F"
    merge_model "$H" /ws/runs/grpo_L/ck/adapter "$L" next_L
    rm -rf "$H"
    merge_model "$L" /ws/runs/grpo_M/ck/adapter "$M" next_M
    rm -rf "$L"
    merge_model "$M" /ws/models/lora-Q-distil2 "$Q" next_Q
    rm -rf "$M"
fi

if [[ ! -f "$Q38/config.json" ]]; then
    echo "=== download official Qwen3.8-27B ==="
    rm -rf "$Q38"
    mkdir -p "$Q38"
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "$RAID":/models \
        -e HOME=/tmp -e HF_HOME=/tmp/hf \
        --entrypoint bash vllm/vllm-openai:nightly-aarch64 \
        -lc "hf download Qwen/Qwen3.8-27B \
            --local-dir /models/Qwen3.8-27B --max-workers 16"
fi

for path in \
    "$RAID/model-GLE" \
    "$RAID/Muse-Glimmer-30B" \
    "$Q" \
    "$BASE" \
    "$RAID/Gemma-4-31B-it" \
    "$Q38"; do
    if [[ ! -f "$path/config.json" ]]; then
        echo "checkpoint preparation failed: $path" >&2
        exit 1
    fi
done

printf '%s\n' \
    "GLE_MODEL=$RAID/model-GLE" \
    "GL_MODEL=$RAID/Muse-Glimmer-30B" \
    "Q_MODEL=$Q" \
    "BASE_MODEL=$BASE" \
    "G4T_MODEL=$RAID/Gemma-4-31B-it" \
    "Q38_MODEL=$Q38" \
    > "$WS/runs/sequential_models.env"
echo "six sequential-search checkpoints are ready"
