"""Measure where wall-clock time goes when verifying one generated kernel.

Verification throughput sets the ceiling on how much training data rejection
sampling can produce. The benchmark harness runs at roughly 6 samples/min on
4 GPUs, which would take days for tens of thousands of candidates, so this
breaks the cost down to find what is actually worth cutting.

Run inside the eval container:
    python3 verify/bench_compile.py
"""

import statistics
import time

import torch
import cuda.tile as ct

TILE = 256


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0


def make_kernel(tile_size: int, depth: int):
    """Build a genuinely distinct kernel.

    The tile shape is a compile-time constant, so varying it forces a real
    compilation rather than a cache hit; depth varies the body size so the
    measurement is not of a single trivial kernel.
    """

    @ct.kernel
    def k(a, b, out):
        i = ct.bid(0)
        x = ct.load(a, index=(i,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
        y = ct.load(b, index=(i,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
        acc = x * y
        for _ in range(depth):
            acc = acc * 1.0001 + x
        ct.store(out, index=(i,), tile=acc)

    return k


def main() -> None:
    print("device:", torch.cuda.get_device_name(0))

    sizes = {"tiny (4096)": 4096, "large (16M)": 16 * 1024 * 1024}
    stream = torch.cuda.current_stream()
    tile_pool = [64, 128, 256, 512, 1024, 2048]

    variant = 0
    for label, n_elem in sizes.items():
        a = torch.rand(n_elem, device="cuda")
        b = torch.rand(n_elem, device="cuda")
        out = torch.empty_like(a)

        first, cached = [], []
        for _ in range(6):
            # Every kernel here is new: distinct tile shape and body depth.
            ts = tile_pool[variant % len(tile_pool)]
            kern = make_kernel(ts, variant)
            variant += 1
            grid = (ct.cdiv(n_elem, ts), 1, 1)

            _, t = timed(ct.launch, stream, grid, kern, (a, b, out))
            first.append(t)
            _, t2 = timed(ct.launch, stream, grid, kern, (a, b, out))
            cached.append(t2)

        print("%-14s JIT compile+launch: median %6.3f s  min %6.3f  max %6.3f"
              % (label, statistics.median(first), min(first), max(first)))
        print("%-14s cached launch     : median %6.4f s"
              % ("", statistics.median(cached)))

    print()
    for label, n_elem in sizes.items():
        a = torch.rand(n_elem, device="cuda")
        ref = a * a
        t0 = time.perf_counter()
        for _ in range(2):
            torch.allclose(ref, a * a, atol=1e-4, rtol=1e-4)
        torch.cuda.synchronize()
        print("%-14s 2x reference+allclose: %6.4f s" % (label, time.perf_counter() - t0))

    print()
    print("Per-candidate verification cost is roughly one JIT compile plus a")
    print("couple of correctness trials. If that is well under a second, the")
    print("benchmark harness's ~10 s/sample is dominated by per-sample process")
    print("startup and the 100 performance trials, not by compilation -- so a")
    print("persistent-worker verifier without timing is the fix.")


if __name__ == "__main__":
    main()
