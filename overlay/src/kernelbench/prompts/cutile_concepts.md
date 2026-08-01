# cuTile Python programming model

cuTile is NVIDIA's tile-based GPU programming model, exposed in Python as the
`cuda.tile` package (conventionally imported as `ct`). It is *not* Triton, CUDA C++,
or CuTe/CUTLASS, and none of their APIs exist here.

Where a SIMT kernel reasons about individual threads, a cuTile kernel reasons about
whole **tiles**: you load a tile of many elements, operate on the entire tile at once,
and store it back. The compiler maps tile operations onto threads, and handles shared
memory, tensor cores and the tensor memory accelerator for you. There are no thread
indices, no `threadIdx`, no explicit masks, and no manual shared-memory staging.

## Kernels and launching

`@ct.kernel` marks a kernel entry point. Kernels run once per block of the launch grid
and cannot be called directly from host code; queue them with `ct.launch`:

```python
import torch
import cuda.tile as ct

TILE = 256

@ct.kernel
def scale_kernel(x, out, factor):
    i = ct.bid(0)                                    # this block's index on axis 0
    t = ct.load(x, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(i,), tile=t * factor)

def scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (ct.cdiv(n, TILE), 1, 1)                  # enough blocks to cover the array
    ct.launch(torch.cuda.current_stream(), grid, scale_kernel,
              (x.view(-1), out.view(-1), factor))
    return out
```

`ct.launch(stream, grid, kernel, args)` takes exactly four positional arguments: a CUDA
stream, a grid tuple of up to 3 dimensions, the kernel object, and a **tuple** of kernel
arguments. `ct.bid(axis)` gives the block index, `ct.num_blocks(axis)` the grid extent.

Array arguments may be any device array exposing DLPack or the CUDA array interface,
so PyTorch tensors are passed directly. Scalars are passed directly too.

## Arrays versus tiles

**Arrays** live in global memory, are mutable, have strided physical layouts, and support
essentially only load/store from within a kernel.

**Tiles** are immutable values that exist only inside kernel code. They support arithmetic,
matmul, reductions, reshaping and so on. Critically:

- **Tile shapes must be compile-time constants, and each dimension must be a power of two.**
  A shape like `(96,)` or a shape read from `x.shape` at runtime will not compile.
- Tile dtypes are compile-time constants (`ct.float32`, `ct.bfloat16`, ...).

## Indexing is in tile space, not element space

`ct.load(array, index, shape)` reads the tile at position `index` in the array's *tile
space*: with `shape=(128,)`, `index=(3,)` reads elements `384:512`. You do not compute
byte or element offsets, and you do not build index vectors as in Triton.

`ct.store(array, index, tile)` is the mirror image.

`Array.tiled_view(tile_shape)` returns a reusable view with `.load(index)` / `.store(index, tile)`
if you prefer not to repeat the shape at each call site.

## Boundary handling

This is the most common source of wrong results. When an array dimension is not a
multiple of the tile shape, the final tile hangs off the edge:

- On **load**, pass `padding_mode` to define the out-of-bounds elements.
  `ct.PaddingMode.ZERO` fills them with zero. The default,
  `ct.PaddingMode.UNDETERMINED`, leaves them unspecified and is only safe when you know
  the tile is fully in bounds.
- On **store**, out-of-bounds writes are silently discarded. No mask is needed.

Choose the padding value to be the identity of whatever you do next. Zero is right for
sums, but for a max reduction you must use `ct.PaddingMode.NEG_INF`, otherwise padding
lanes contribute a spurious 0:

```python
@ct.kernel
def row_max_kernel(x, out):
    i = ct.bid(0)
    t = ct.load(x, index=(i, 0), shape=(1, COLS), padding_mode=ct.PaddingMode.NEG_INF)
    ct.store(out, index=(i,), tile=ct.max(t, axis=1))
```

## Reductions

`ct.sum`, `ct.max`, `ct.min`, `ct.prod`, `ct.argmax`, `ct.argmin` take `axis` and
`keepdims`, following NumPy semantics; `axis=None` reduces to a scalar. `ct.reduce` and
`ct.scan` accept a custom binary function plus an identity element. Broadcasting between
tiles follows NumPy rules, so a `(M, 1)` reduction result combines directly with an
`(M, N)` tile.

## Matrix multiply

`ct.mma(a, b, acc)` computes `a @ b + acc` on tensor cores. The idiom is to accumulate in
FP32 regardless of input precision, loop over K-tiles, zero-pad partial K-tiles, and cast
on store. Partial M/N edge tiles need no special handling because stores discard
out-of-bounds writes:

```python
TM, TN, TK = 64, 64, 32

@ct.kernel
def gemm_kernel(A, B, C):
    bx, by = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(TM, TK))):
        a = ct.load(A, index=(bx, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, by), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bx, by), tile=acc.astype(C.dtype))
```

`ct.matmul(a, b)` and the `@` operator perform a plain matmul without accumulation.
`ct.num_tiles(array, axis, shape)` gives the number of tiles along an axis, which is how
you bound a K-loop.

## Control flow and compile-time values

Ordinary Python `if`/`for`/`while` work inside kernels. Loop bounds and conditions may be
runtime values, but anything that determines a tile's shape or dtype must be a
compile-time constant. Python scalars closed over from module scope are compile-time
constants, which is why tile sizes are written as module-level literals.
