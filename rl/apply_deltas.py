"""Fold a GRPO checkpoint's trained tensors back into a full adapter.

grpo.py freezes the expert parameters and checkpoints only what it trains, which
is 137 MB instead of 27.5 GB and is what makes checkpointing every iteration
affordable. The cost is that a checkpoint is not a loadable adapter on its own:
it has to be recombined with the adapter the run started from before anything can
serve or merge it.

Usage:
    python3 rl/apply_deltas.py --adapter models/lora-C-speed \\
        --deltas runs/grpo/ck/trainable.pt --out models/lora-D-grpo
"""

import argparse
import json
import os
import shutil
import struct


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Adapter the run started from.")
    ap.add_argument("--deltas", required=True, help="trainable.pt from grpo.py.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src = os.path.join(args.adapter, "adapter_model.safetensors")
    print("loading %s" % src)
    tensors = load_file(src)
    print("  %d tensors" % len(tensors))

    deltas = torch.load(args.deltas, map_location="cpu")
    print("checkpoint has %d trained tensors" % len(deltas))

    # Both sides carry the same base_model.model.model prefix. The only
    # difference is the active adapter's name, which a live state_dict includes
    # and a saved adapter does not:
    #   live:  ...lora_A.default.weight
    #   saved: ...lora_A.weight
    def normalise(k):
        return k.replace(".default.weight", ".weight")

    applied = missing = 0
    for k, v in deltas.items():
        nk = normalise(k)
        if nk in tensors:
            if tensors[nk].shape != v.shape:
                raise SystemExit("shape mismatch for %s: adapter %s vs checkpoint %s"
                                 % (nk, tuple(tensors[nk].shape), tuple(v.shape)))
            tensors[nk] = v.to(tensors[nk].dtype)
            applied += 1
        else:
            missing += 1
            if missing <= 5:
                print("  no match in adapter for %s" % nk)

    if applied == 0:
        raise SystemExit("no checkpoint tensor matched the adapter; key naming "
                         "differs and the merge would silently do nothing")
    print("applied %d, unmatched %d" % (applied, missing))

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "adapter_model.safetensors"),
              metadata={"format": "pt"})
    for f in os.listdir(args.adapter):
        if f != "adapter_model.safetensors":
            shutil.copy(os.path.join(args.adapter, f), os.path.join(args.out, f))

    hdr = read_safetensors_header(os.path.join(args.out, "adapter_model.safetensors"))
    print("wrote %s (%d tensors)" % (args.out, len(hdr) - ("__metadata__" in hdr)))


if __name__ == "__main__":
    main()
