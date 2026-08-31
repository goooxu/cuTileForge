#!/usr/bin/env python3
"""Create a lightweight LoRA view with a scaled effective delta.

PEFT applies ``lora_alpha / r * B @ A``.  Keeping the tensors unchanged and
scaling ``lora_alpha`` therefore gives exactly

    W(scale) = W_base + scale * (W_adapter - W_base)

after merge.  The large safetensors file is linked relatively so the candidate
adapter costs only a config file on the shared workspace.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", required=True, type=float)
    args = ap.parse_args()

    if not 0.0 < args.scale <= 1.0:
        raise SystemExit("--scale must be in (0, 1]")

    source = Path(args.adapter).resolve()
    out = Path(args.out).resolve()
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise SystemExit("not a PEFT adapter: %s" % source)

    config = json.loads(config_path.read_text())
    original_alpha = float(config["lora_alpha"])
    config["lora_alpha"] = original_alpha * args.scale

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")

    relative_weights = os.path.relpath(weights_path, out)
    (out / "adapter_model.safetensors").symlink_to(relative_weights)
    metadata = {
        "source_adapter": str(source),
        "scale": args.scale,
        "original_lora_alpha": original_alpha,
        "scaled_lora_alpha": config["lora_alpha"],
        "effective_multiplier": args.scale,
    }
    (out / "scale.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("wrote %s" % out)
    print("lora_alpha %.8g -> %.8g (delta %.4fx)"
          % (original_alpha, config["lora_alpha"], args.scale))
    print("weights -> %s" % relative_weights)


if __name__ == "__main__":
    main()
