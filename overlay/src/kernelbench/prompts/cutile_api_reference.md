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

# cuTile Python (`cuda.tile`) API reference

Generated from cuda-tile 1.4.0. This is the complete set of public operations; anything not listed here does not exist in cuTile.

## Kernel definition and launch

#### `ct.kernel(function=None, /, **kwargs)`

A *tile kernel* is a function executed by each block in a grid.

Functions with this decorator are kernels.

Kernels are the entry points of tile code.
Their execution space shall be only tile code; they cannot be called from host code.

Kernels cannot be called directly. Instead, use `launch` to
queue a kernel for execution over a grid.

The types usable as parameters to a kernel are described in the data model.

Args:
    num_ctas: Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
        Default: None (auto).
    occupancy: Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
    opt_level: Optimization level [0, 3], default 3.
    num_worker_warps: Number of warps in the CUDA core warp groups in a
        warp-specialized kernel. The compiler may add warps
        (e.g., for asynchronous memory transfers) that are not counted here. ...

#### `ct.function(func=None, /, *, host=False, tile=True)`

*Tile functions* are functions that are usable in tile code.

This decorator indicates what execution spaces a function can be called from.
With no arguments, it denotes a tile-only function.

When an unannotated function is called by a tile function, tile shall be added to the
unannotated function's execution space.
This process is recursive.
No explicit annotation is required.

The types usable as parameters to a tile function are described in the data model.

Args:
    host (bool, optional): Whether the function can be called from host code.
        Default is False.
    tile (bool, optional): Whether the function can be called from tile code.
        Default is True.

#### `ct.launch(stream, grid, kernel, kernel_args, /)`

Launch a cuTile kernel.

Args:
   stream: The CUDA stream to execute the kernel on.
   grid: Tuple of up to 3 grid dimensions to execute the kernel over.
   kernel: The kernel to execute.
   kernel_args: Positional arguments to pass to the kernel.

#### `ct.bid(axis) -> 'int'`

Gets the index of current block.

Args:
    axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

Returns:
    int32:

#### `ct.num_blocks(axis) -> 'int'`

Gets the number of blocks along the axis.

Args:
    axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

Returns:
    int32:

#### `ct.num_tiles(array: 'Array', /, axis: 'int', shape: 'Constant[Shape]', order: 'Constant[Order]' = 'C') -> 'int'`

Gets the number of tiles in the tile space of the array along the `axis`.

Args:
    array (Array): An array object on a cuda device.
    axis (const int): The axis of the tile partition space to get the dim size.
    shape (const int...): A sequence of const integers definining the shape of the tile.
    order ("C" or "F", or tuple[const int,...]): Order of axis mapping. See `load`.

Returns:
    int32

#### `ct.cdiv(x, y, /) -> 'TileOrScalar'`

Computes ceil(x / y). Can be used on the host.

Args:
    x (Tile): int tile.
    y (Tile): int tile.

Returns:
    Tile:

## Loading and storing

#### `ct.load(array: 'Array', /, index: 'Shape', shape: 'Constant[Shape]', *, order: 'Constant[Order]' = 'C', padding_mode: 'PaddingMode' = <PaddingMode.UNDETERMINED: 'undetermined'>, latency: 'Optional[int]' = None, allow_tma: 'Optional[bool]' = None, memory_order: 'MemoryOrder' = <MemoryOrder.WEAK: 'weak'>, memory_scope: 'MemoryScope' = <MemoryScope.NONE: 'none'>) -> 'Tile'`

Loads a tile from the `array` which is partitioned into a tile space.

The tile space is the result of partitioning the `array` into a grid of equally
sized tiles specified by `shape`.

For example, partitoning a 2D `array` of shape `(M, N)` using tile shape
`(tm, tn)` results in a 2D tile space of size `(cdiv(M, tm), cdiv(N, tn))`.
An index into this tile space using index `(i, j)` produces a tile of size `(tm, tn)`::

    t = ct.load(array, (i, j), (tm, tn))  # `t` has shape (tm, tn)

The result tile `t` will be computed according to ::

    t[x, y] = array[i * tm + x, j * tn + y]  (for all 0<=x<tm, 0<=y<tn)

For a tile that partially extends beyond the array boundaries, out-of-bound elements
are filled according to `padding_mode`.
If the tile lies entirely outside the array, the behavior is undefined. ...

#### `ct.store(array: 'Array', /, index: 'Shape', tile: 'TileOrScalar', *, order: 'Constant[Order]' = 'C', latency: 'Optional[int]' = None, allow_tma: 'Optional[bool]' = None, memory_order: 'MemoryOrder' = <MemoryOrder.WEAK: 'weak'>, memory_scope: 'MemoryScope' = <MemoryScope.NONE: 'none'>) -> 'None'`

Stores a `tile` value into the `array` at the `index` of its tile space.

The tile space is the result of partitioning the `array` into a grid of tiles
with equal size defined by the shape of the `tile`.

For example, given a tile `t` of shape `(tm, tn)` and array of shape `(M, N)`::

    # tile `t` has shape (tm, tn)
    ct.store(array, (i, j), t)

The above call to `store` will store elements according to::

    array[i * tm + x, i * tn + y] = t[x, y]  (for 0<=x<tm, 0<=y<tn)

For a tile that partially extends beyond the array boundaries, out-of-bound elements
are ignored.
If the tile lies entirely outside the array, the behavior is undefined.

Args:
    array (Array): The array to store to.
    index (tuple[int,...]): An index in the tile space of `array`.
        `shape` is inferred from the `tile` argument. ...

#### `ct.gather(array, indices, /, *, mask=None, padding_value=0, check_bounds=True, latency=None) -> 'Tile'`

Loads a tile from the `array` elements specified by `indices`.

`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.

The result shape will be the same as the broadcasted shape of indices.

For example, consider a 2-dimensional array. In this case, indices must be a tuple
of length 2. Suppose that `ind0` and `ind1` are integer tiles
of shapes `(M, N, 1)` and `(M, 1, K)`.
Then the result tile will have the broadcasted shape `(M, N, K)`::

    t = ct.gather(array, (ind0, ind1))   # `t` has shape (M, N, K)

The result tile `t` will be computed according to ::

    t[i, j, k] = array[ind0[i, j, 0], ind1[i, 0, k]]   (for all 0<=i<M, 0<=j<N, 0<=k<K)

If the array is 1-dimensional, `indices` can be passed as a tile rather than a tuple. ...

#### `ct.scatter(array, indices, value, /, *, mask=None, check_bounds=True, latency=None)`

Stores a tile `value` into the `array` elements specified by `indices`.

`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.

`value` must be a scalar or a tile whose shape is broadcastable to the
common shape of `indices`.

For example, consider a 2-dimensional array. In this case, indices must be a tuple
of length 2. Suppose that `ind0` and `ind1` are integer tiles
of shapes `(M, N, 1)` and `(M, 1, K)`, and `value` is a tile of shape of `(N, K)`::

    # ind0: (M, N, 1),  ind1: (M, 1, K),  value: (N, K)
    ct.scatter(array, (ind0, ind1), value)

The above call to `scatter` will store elements according to ::

    array[ind0[i, j, 0], ind1[i, 0, k]] = value[j, k] ...

#### `ct.load_advanced_indexing(array: 'Array', indices, /, *, padding_mode: 'PaddingMode' = <PaddingMode.UNDETERMINED: 'undetermined'>, latency: 'Optional[int]' = None, allow_tma: 'Optional[bool]' = None) -> 'Tile'`

Loads a tile from non-contiguous slices of `array`.

`indices` is a tuple of length `array.ndim`.  Exactly one entry must
be a 1-D integer :class:`Tile` (the *sparse dim*); every other entry must
be a :class:`Slice` `(start, length)` where `start` is a runtime
element-space offset and `length` is a compile-time power-of-two tile
size.

The sparse-dim tile contains element-space indices — each value selects
one slice of the array along that dimension.  Each dense-dim
:class:`Slice` describes a contiguous range `[start, start + length)`.
The resulting tile has shape `(len_0, ..., len_{n-1})` where `len_i`
is the index-tile length for the sparse dim or `Slice.length` for dense
dims.

If the tile lies entirely outside the tiled view, the behavior is undefined.

Args:
    array (Array): Array to load from.
    indices (tuple): Length must equal `array.ndim`.  Exactly one entry ...

#### `ct.store_advanced_indexing(array: 'Array', indices, tile: 'TileOrScalar', /, *, latency: 'Optional[int]' = None, allow_tma: 'Optional[bool]' = None) -> 'None'`

Stores a `tile` into non-contiguous slices of `array`.

Uses the same `indices` convention as :func:`load_advanced_indexing` — exactly
one entry is a 1-D integer :class:`Tile` (sparse dim) and the rest are
:class:`Slice` objects (dense dims).
The tile's shape must exactly match the shape implied by the indices.

If the tile lies entirely outside the tiled view, the behavior is undefined.

Args:
    array (Array): Array to store into.
    indices (tuple): Same convention as :func:`load_advanced_indexing`.
    tile (Tile): Tile to store. Shape must exactly match the shape
        implied by `indices`.
    latency (int, optional): DRAM traffic hint (1 = low, 10 = high).
    allow_tma (bool, optional): If `False`, TMA will not be used.

## Tile creation

#### `ct.zeros(shape, dtype) -> 'Tile'`

Creates a tile filled with zeros.

Args:
    shape (tuple[const int,...]):  The shape of the tile.
    dtype (DType): The Data type of the tile.

Returns:
    Tile:

#### `ct.ones(shape, dtype) -> 'Tile'`

Creates a tile filled with ones.

Args:
    shape (tuple[const int,...]):  The shape of the tile.
    dtype (DType): The Data type of the tile.

Returns:
    Tile:

#### `ct.full(shape: 'Shape', fill_value: 'Scalar', dtype: 'DType') -> 'Tile'`

Creates a tile filled with given value.

Args:
    shape (tuple[const int,...]):  The shape of the tile.
    fill_value (int  float  bool]): Value for the tile.
    dtype (DType): The Data type of the tile.

Returns:
    Tile:

#### `ct.arange(size, /, *, dtype) -> 'Tile'`

Creates a tile with value starting from 0 to `size - 1`.

Args:
    size (const int): Size of the tile.
    dtype (DType): Datatype of the tile.

Returns:
    Tile:

#### `ct.astile(value, /, *, dtype: 'DType') -> 'Tile'`

Creates a tile from a value.

Args:
    value (scalar | (nested) tuple of scalar): A scalar (yielding a 0-d tile),
        or a (possibly nested) tuple of scalars whose nesting determines the 
        tile's shape. Every tuple's length must be a power of two, and sibling tuples
        at each level must have uniform length.
    dtype (DType): The Data type of the tile.

Returns:
    Tile: A tile shaped from `value`, with elements cast to `dtype`.

## Matrix multiply

#### `ct.mma(x, y, /, acc, *, use_fast_acc: 'bool' = False) -> 'Tile'`

Matrix multiply-accumulate.

Computes `(x @ y) + acc` as a single operation
(where `@` denotes matrix multiplication).
Preserves the dtype of `acc`.

Args:
    x (Tile): LHS of the mma, 2D or 3D.
    y (Tile): RHS of the mma, 2D or 3D.
    acc (Tile): Accumulator of mma.
    use_fast_acc (bool): Enable fast accumulation mode, which trades accumulator
        precision for throughput. Requires fp8 input dtypes
        (`float8_e4m3fn` or `float8_e5m2`). Currently only has an effect on
        Hopper GPUs; silently ignored on other architectures. Default: `False`
        (since CTK 13.3).

Supported datatypes:

+----------+---------------+
 Input      Acc/Output   |
+==========+===============+
 f16        f16 or f32   |
+----------+---------------+
 bf16       f32          |
+----------+---------------+
 f32        f32          |
+----------+---------------+
 f64        f64          | ...

#### `ct.matmul(x, y, /) -> 'Tile'`

Performs matrix multiply on the given tiles.

Args:
    x (Tile): LHS of the matmul, 1D, 2D, or 3D.
    y (Tile): RHS of the matmul, 1D, 2D, or 3D.

Supported input datatypes: [f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8]

If `x` and `y` have different dtype, they will first be promoted to common
dtype. The result dtype is the same as the promoted input types.
Shape of `x` and `y` will be broadcasted to up until the last two axes.

Returns:
    Tile:

#### `ct.mma_scaled(x, x_scale, y, y_scale, /, acc) -> 'Tile'`

Block-scaled matrix multiply-accumulate.

Computes a matrix multiply-accumulate where inputs are scaled by block scales
along the K dimension before the mma::

    result[i, j] = sum(x[i, k] * x_scale[i, k // B] * y[k, j] * y_scale[k // B, j]
                       for k in range(K)) + acc[i, j]

The scaling block size is `B = K // K_s`, where `K_s` is the K dimension of the scale tile.
`K` must be divisible by `K_s`, and `B` must be one of the allowed values listed
in the table below.

Args:
    x (Tile): LHS input, 2D or 3D `[..., M, K]`.
    x_scale (Tile): Scale factors for x, shape `[..., M, K_s]`.
        All dimensions except K_s must match x exactly.
    y (Tile): RHS input, 2D or 3D `[..., K, N]`.
    y_scale (Tile): Scale factors for y, shape `[..., K_s, N]`.
        All dimensions except K_s must match y exactly.
    acc (Tile): Accumulator `[..., M, N]`. ...

## Reduction and scan

#### `ct.sum(x, /, axis=None, *, keepdims=False, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs sum reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.
        rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
        flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.prod(x, /, axis=None, *, keepdims=False, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs prod reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.
        rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
        flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.max(x, /, axis=None, *, keepdims=False, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs max reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.
        flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.min(x, /, axis=None, *, keepdims=False, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs min reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.
        flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.argmax(x, /, axis=None, *, keepdims=False) -> 'Tile'`

Performs argmax reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.

Returns:
    Tile:

#### `ct.argmin(x, /, axis=None, *, keepdims=False) -> 'Tile'`

Performs argmin reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None  const int  tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.

Returns:
    Tile:

#### `ct.reduce(x, /, axis, func, identity, *, keepdims=False)`

Apply custom reduction function along axis.

Args:
    x: input tile or a tuple of tiles to be reduced. If a tuple is provided, shapes
        of the tiles in the tuple must be broadcastable to a common shape.
    axis (int): an integer constant that specifies the axis to reduce along.
    func: function for combining two values. If `x` is a single tile, then the function
        must take two 0d tile arguments and return the combined 0d tile. For example,
        `lambda a, b: a + b` or `operator.add` can be used to implement the sum reduction.
        If `x` is a tuple of N tiles, then the function takes 2N tiles and returns a tuple
        of N combined tiles. The first N arguments correspond to one of the groups of values
        being combined, while the rest correspond to the other.
    identity: a constant scalar or a tuple of constant scalars that specifies the identity ...

#### `ct.scan(x, /, axis, func, identity, *, reverse=False)`

Apply custom scan (inclusive prefix) function along axis.

Args:
    x: input tile or a tuple of tiles to be scanned. If a tuple is provided, shapes
        of the tiles in the tuple must be broadcastable to a common shape.
    axis (int): an integer constant that specifies the axis to scan along.
    func: function for combining two values. If `x` is a single tile, then the function
        must take two 0d tile arguments and return the combined 0d tile. For example,
        `lambda a, b: a + b` or `operator.add` can be used to implement cumsum.
        If `x` is a tuple of N tiles, then the function takes 2N tiles and returns a tuple
        of N combined tiles. The first N arguments correspond to one of the groups of values
        being combined, while the rest correspond to the other.
    identity: a constant scalar or a tuple of constant scalars that specifies the identity ...

#### `ct.cumsum(x, /, axis=0, *, reverse=False, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs cumsum on tile along the `axis`.

Args:
    x (Tile): input tile
    axis (const int): the axis for scan, default 0.
    reverse (const bool): if True, the scan is performed in the reverse direction.
    rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
    flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.cumprod(x, /, axis=0, *, reverse=False, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'Tile'`

Performs cumprod on tile along the `axis`.

Args:
    x (Tile): input tile
    axis (const int): the axis for scan, default 0.
    reverse (const bool): if True, the scan is performed in the reverse direction.
    rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
    flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

## Shape and dtype

#### `ct.reshape(x, /, shape) -> 'Tile'`

Reshapes a tile to the specified shape.

One of the shape elements may be specified as -1 to indicate that the
corresponding dimension is to be inferred automatically.

For example, reshaping a `(16, 2)` tile to `(8, -1)` will
produce a tile of shape `(8, 4)`: as there are 32 elements in total,
the second dimension will be computed as 32 divided by 8.

Args:
    x (Tile): input tile.
    shape (tuple[const int,...]): target shape.

Returns:
    Tile:

#### `ct.transpose(x, /, axis0=None, axis1=None) -> 'Tile'`

Transposes two axes of the input tile with at least 2 dimensions.

For a 2-dimensional tile, the two axes are transposed if `axis0` and `axis1` are not specified.
For tiles with more than 2 dimensions, `axis0` and `axis1` must be explicitly specified.

Args:
    x (Tile): input tile.
    axis0 (const int): the first axis to transpose.
    axis1 (const int): the second axis to transpose.

Returns:
    Tile:

#### `ct.permute(x, /, axes) -> 'Tile'`

Permutes the axes of the input tile.

Args:
    x (Tile): input tile.
    axes (tuple[const int,...]): the desired axes order.

Returns:
    Tile:

#### `ct.expand_dims(x, /, axis) -> 'Tile'`

Reshapes the tile by inserting a new axis of size 1 at given position.

This can also be done via the NumPy-style syntax: `x[:, None]` or `x[np.newaxis, :]`

Args:
    x (Tile): input tile.
    axis (const int): axis to expand the tile dimension.

Returns:
    Tile:

#### `ct.broadcast_to(x, /, shape) -> 'Tile'`

Broadcasts a tile to the specified shape
following Numpy broadcasting rule.

Args:
    x (Tile): input tile.
    shape (tuple[const int,...]): target shape.

Returns:
    Tile:

#### `ct.cat(tiles, /, axis) -> 'Tile'`

Concatenates two tiles along the `axis`.

Args:
    tiles (tuple): a pair of tiles to concatenate.
    axis (const int): axis to concatenate the tiles.

Returns:
    Tile:

Notes:
    Due to power-of-two assumption on all tile shapes,
    the two input tiles must have the same shape.

#### `ct.extract(x, /, index, shape) -> 'Tile'`

Extracts a smaller tile from input tile.

Partition the input tile into a grid with subtile shape
and return a tile given the index into the grid. Similar
to `load` but performed on a tile.

Args:
    x (Tile): input tile.
    index (Shape): Index into the grid of subtiles, not element index.
        Each dimension `i` has `x.shape[i] // shape[i]` subtiles;
        valid values are `[0, x.shape[i] // shape[i])`.
        For example, extracting shape `(4,)` from a `(128,)` tile
        gives 32 subtiles, so valid indices are 0–31.
    shape (Shape): The shape of the extracted tile. Must evenly divide
        `x.shape` in every dimension.

Returns:
    Tile:

#### `ct.astype(x, dtype, /) -> 'Tile'`

Converts a tile to the specified data type.

Args:
    x (Tile): input tile.
    dtype (DType): target data type.

Returns:
    Tile:

#### `ct.bitcast(x, /, dtype) -> 'Tile'`

Reinterpets tile as being of specified data type.

Args:
    x (Tile): input tile.
    dtype (DType): target data type.

Returns:
    Tile:

## Elementwise math

#### `ct.abs(x, /) -> 'TileOrScalar'`

Perform `abs` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.add(x, y, /, *, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise add on two tiles.

Can also use builtin operation `x + y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.sub(x, y, /, *, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise sub on two tiles.

Can also use builtin operation `x - y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.mul(x, y, /, *, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise mul on two tiles.

Can also use builtin operation `x * y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.truediv(x, y, /, *, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise truediv on two tiles.

Can also use builtin operation `x / y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.floordiv(x, y, /) -> 'TileOrScalar'`

Elementwise floordiv on two tiles.

Can also use builtin operation `x // y`.

Supports both integer and floating-point operands. For float inputs,
the result is `floor(x / y)` as a float (e.g. `5.5 // 2.2 == 2.0`).

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.mod(x, y, /) -> 'TileOrScalar'`

Elementwise mod on two tiles.

Can also use builtin operation `x % y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.negative(x, /) -> 'TileOrScalar'`

Same as `-x`.

Args:
    x (Tile): input tile.

Returns:
    Tile:

#### `ct.pow(x, y, /) -> 'TileOrScalar'`

Elementwise pow on two tiles.

Can also use builtin operation `x ** y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.exp(x, /, *, rounding_mode: 'Optional[RoundingMode]' = None) -> 'TileOrScalar'`

Perform `exp` on a tile.

Args:
    x (Tile):
    rounding_mode (RoundingMode): Supported values:

        - `RoundingMode.FULL` (f32 only)
        - `RoundingMode.APPROX` (f32 only)

        (since CTK 13.3)

Returns:
    Tile:

#### `ct.exp2(x, /, *, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Perform `exp2` on a tile.

Args:
    x (Tile):
    flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.log(x, /) -> 'TileOrScalar'`

Perform `log` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.log2(x, /) -> 'TileOrScalar'`

Perform `log2` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.sqrt(x, /, *, rounding_mode: 'Optional[RoundingMode]' = None, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Perform `sqrt` on a tile.

Args:
    x (Tile):
    rounding_mode (RoundingMode): The rounding mode for the operation, only supported for float types, default is RoundingMode.RN when applicable.
    flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.rsqrt(x, /, *, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Perform `rsqrt` on a tile.

Args:
    x (Tile):
    flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

Returns:
    Tile:

#### `ct.ceil(x, /) -> 'TileOrScalar'`

Perform `ceil` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.floor(x, /) -> 'TileOrScalar'`

Perform `floor` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.sin(x, /) -> 'TileOrScalar'`

Perform `sin` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.cos(x, /) -> 'TileOrScalar'`

Perform `cos` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.tan(x, /) -> 'TileOrScalar'`

Perform `tan` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.sinh(x, /) -> 'TileOrScalar'`

Perform `sinh` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.cosh(x, /) -> 'TileOrScalar'`

Perform `cosh` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

#### `ct.tanh(x, /, *, rounding_mode: 'Optional[RoundingMode]' = None) -> 'TileOrScalar'`

Perform `tanh` on a tile.

Args:
    x (Tile):
    rounding_mode (RoundingMode): Supported values:

        - `RoundingMode.FULL` (f32 only)
        - `RoundingMode.APPROX` (f32 only)

        (since CTK 13.2)

Returns:
    Tile:

#### `ct.atan2(x1, x2, /) -> 'TileOrScalar'`

Elementwise atan2 of two tiles.

Computes the element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

Args:
    x1 (Tile): Numerator tile (y-coordinate).
    x2 (Tile): Denominator tile (x-coordinate).

The `shape` of `x1` and `x2` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile: The angles in radians, in the range [-pi, pi].

#### `ct.maximum(x, y, /, *, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise maximum on two tiles.

Can also use builtin operation `max(x, y)`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.minimum(x, y, /, *, flush_to_zero: 'bool' = False) -> 'TileOrScalar'`

Elementwise minimum on two tiles.

Can also use builtin operation `min(x, y)`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.
            flush_to_zero (const bool): If True, flushes subnormal inputs and results to sign-preserving zero, default is False.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.isnan(x, /) -> 'TileOrScalar'`

Perform `isnan` on a tile.

Args:
    x (Tile):

Returns:
    Tile:

## Comparison and selection

#### `ct.where(cond, x, y, /) -> 'Tile'`

Returns elements chosen from x or y depending on condition.

Args:
    cond (Tile): Boolean tile of shape `S`.
    x (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is True.
    y (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is False.

Returns:
    Tile:

#### `ct.equal(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `==`.

Can also use builtin operation `x == y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.not_equal(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `!=`.

Can also use builtin operation `x != y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.less(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `<`.

Can also use builtin operation `x < y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.less_equal(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `<=`.

Can also use builtin operation `x <= y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.greater(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `>`.

Can also use builtin operation `x > y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.greater_equal(x, y, /) -> 'TileOrScalar'`

Compare two tiles elementwise with `>=`.

Can also use builtin operation `x >= y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

## Bitwise

#### `ct.bitwise_and(x, y, /) -> 'TileOrScalar'`

Elementwise bitwise_and on two tiles.

Can also use builtin operation `x & y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.bitwise_or(x, y, /) -> 'TileOrScalar'`

Elementwise bitwise_or on two tiles.

Can also use builtin operation `x | y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.bitwise_xor(x, y, /) -> 'TileOrScalar'`

Elementwise bitwise_xor on two tiles.

Can also use builtin operation `x ^ y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.bitwise_not(x, /) -> 'TileOrScalar'`

Elementwise bitwise not on a tile.

Can also use builtin operator `~x`.

Args:
    x (Tile): input tile.

Returns:
    Tile:

#### `ct.bitwise_lshift(x, y, /) -> 'TileOrScalar'`

Elementwise bitwise_lshift on two tiles.

Can also use builtin operation `x << y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

#### `ct.bitwise_rshift(x, y, /) -> 'TileOrScalar'`

Elementwise bitwise_rshift on two tiles.

Can also use builtin operation `x >> y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

## Atomics

#### `ct.atomic_add(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'Tile'`

Bulk atomic post-increment of array elements at given indices.

For each specified index, `atomic_add()` reads the corresponding array element,
adds `update` to it, and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_add()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile
rather than a tuple of length 1. ...

#### `ct.atomic_max(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'TileOrScalar'`

Bulk atomic maximum value assignment on array elements at given indices.

For each specified index, `atomic_max()` reads the corresponding array element,
computes the maximum between its value and the corresponding value of `update`,
and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_max()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile ...

#### `ct.atomic_min(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'TileOrScalar'`

Bulk atomic minimum value assignment on array elements at given indices.

For each specified index, `atomic_min()` reads the corresponding array element,
computes the minimum between its value and the corresponding value of `update`,
and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_min()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile ...

#### `ct.atomic_and(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'TileOrScalar'`

Bulk atomic AND operation on array elements at given indices.

For each specified index, `atomic_and()` reads the corresponding array element,
computes the bitwise AND between its value and the corresponding value of `update`,
and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_and()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile
rather than a tuple of length 1. ...

#### `ct.atomic_or(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'Tile'`

Bulk atomic OR operation on array elements at given indices.

For each specified index, `atomic_or()` reads the corresponding array element,
computes the bitwise OR between its value and the corresponding value of `update`,
and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_or()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile
rather than a tuple of length 1. ...

#### `ct.atomic_xor(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'Tile'`

Bulk atomic XOR operation on array elements at given indices.

For each specified index, `atomic_xor()` reads the corresponding array element,
computes the bitwise XOR between its value and the corresponding value of `update`,
and writes the modified value back to the same location.
The original value of the element before the update is returned.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_xor()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile
rather than a tuple of length 1. ...

#### `ct.atomic_cas(array, indices, expected, desired, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'Tile'`

Bulk atomic compare-and-swap on array elements with given indices.

For each specified index, `atomic_cas()` compares the corresponding array element
to the `expected` value. If it matches, it is then overwritten with the `desired` value;
otherwise, no update is performed. In either case, the old value of the element is returned.
For each individual element, the described compare-and-swap operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual updates is unspecified.

`atomic_cas()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile ...

#### `ct.atomic_xchg(array, indices, update, /, *, check_bounds=True, memory_order=<MemoryOrder.ACQ_REL: 'acq_rel'>, memory_scope=<MemoryScope.DEVICE: 'device'>) -> 'Tile'`

Bulk atomic exchange of array elements at given indices.

For each specified index, `atomic_xchg()` stores the corresponding `update`
to the array element at that location, and returns the original value of the element
before the update.

For each individual element, the operation is performed atomically,
but the operation as a whole is not atomic, and the order of individual writes is unspecified.

`atomic_xchg()` follows the same convention as `gather()` and `scatter()`:
`indices` must be a tuple whose length equals the `array` rank.
All elements of this tuple must be integer tiles or scalars of the same shape,
or different shapes that are broadcastable to a common shape.
If the array is 1-dimensional, `indices` can be passed as a single tile
rather than a tuple of length 1.

`update` must be a scalar or a tile whose shape is broadcastable to the
common shape of `indices`. ...

## Metaprogramming and debugging

#### `ct.static_eval(expr, /)`

Evaluates the given Python expression at compile time.

The expression is evaluated using standard Python semantics, not Tile
semantics. It can reference global variables and local variables from
the surrounding tile function.

If a referenced variable is a compile-time constant value, it will be represented
with a corresponding Python object of that value. For example, a constant integer 3 will
be passed as a plain `int` object of value 3.

If a referenced variable has dynamic value, such as a tile or an array,
it will be passed as a proxy object that allows querying compile-time attributes.
For example, if `x` is a tile, one can use `x.shape` to obtain the tile shape
as a tuple of integers.

The expression is allowed to return a proxy object for a dynamic value.
This can be used to select one of multiple dynamic values based on a compile-time ...

#### `ct.static_iter(iterable)`

Iterates at compile time.

Can only be used as the iterable of a `for` loop::

    for ... in ct.static_iter(...):
        ...

The surrounded expression is evaluated using the same rules as `static_eval`:
it can reference global and local variables, and use the full Python syntax,
but must not perform any run-time operations.

The expression must return a Python iterable, whose length must not exceed some
pre-defined number of iterations (currently, 1000). Before any further processing is done,
the contents of the iterable are saved to a temporary list, and each item is checked
to be valid, as if it were a result of a `static_eval` expression
(i.e., it must be a supported compile-time constant value or a proxy object
for a dynamic value such as a tile).

Finally, for each item of the iterable, the loop body is inlined, with the induction variable(s) ...

#### `ct.static_assert(condition, message=None, /)`

Asserts that a condition is true at compile time.

First, `condition` is evaluated using the same rules as `static_eval`:
it can reference global and local variables, and use the full
Python syntax, but must not perform any run-time operations.

The `condition` must evaluate to a compile-time constant boolean.
If it evaluates to `True`, compilation continues normally,
and the `message` expression is not evaluated.

If `condition` evaluates to `False`, then the `message` expression is evaluated using
the `static_eval` semantics. If the result of the evaluation is None,
it is replaced with an empty string. Otherwise, it is converted to a string using
the builtin `str()` function. Then, a `TileStaticAssertionError` is raised
with the evaluated message string.

Because `message` is evaluated using the `static_eval` semantics, ...

#### `ct.assert_(cond, /, message=None) -> 'None'`

Assert that all elements of the given tile are True.

Args:
    cond (Tile): Boolean tile.
    message (str): Message to print if condition is false.

Notes:
    This operation has significant overhead, and should only be used
    for debugging purpose.

#### `ct.print(*args, sep: 'str' = ' ', end: 'str' = '\n') -> 'None'`

Print values at runtime from the device using Python-style syntax.

Supports Python f-strings and positional arguments similar to Python's
built-in `print()` function.

Args:
    *args: Values to print. Each argument can be:
        - A string literal or f-string
        - A tile value (format inferred from dtype: int→`%d`, float→`%f`)
    sep (str): Separator inserted between arguments (default: `' '`)
    end (str): String appended after the last argument (default: `'\n'`)

#### `ct.printf(format, *args) -> 'None'`

Print the values at runtime from the device

Args:
    format (str): a c-printf style format string
        in the form of `%[flags][width][.precision][length]specifier`,
        where specifier is limited to integer and float for now, i.e.
        `[diuoxXeEfFgGaA]`

    *args (tuple[Tile, ...]):
        Only tile input is supported.

#### `ct.pack_to_bytes(x, /) -> 'Tile'`

Flattens a tile and reinterprets its raw bytes as uint8 elements.

The total number of bits of the input tile must be divisible by 8.

Args:
    x (Tile): input tile.

Returns:
    Tile: a 1D uint8 tile with `total_elements * bit width // 8` elements.

#### `ct.unpack_from_bytes(x, /, dtype) -> 'Tile'`

Reinterprets a 1D uint8 byte tile as a 1D tile of the target data type.

The inverse of `pack_to_bytes`. The input must be a 1D tile of
dtype uint8, and the total number of bits must be divisible by the
target data type bit width.

Args:
    x (Tile): a 1D tile of dtype uint8.
    dtype (DType): target data type.

Returns:
    Tile: a 1D tile of `dtype` with `num_bytes * 8 // bit width` elements.

## Data types

Referenced as `ct.<name>`: `float64`, `float32`, `float16`, `bfloat16`, `tfloat32`, `float8_e4m3fn`, `float8_e5m2`, `float8_e8m0fnu`, `float4_e2m1fn`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool_`.

## Enums and classes

- `ct.PaddingMode`: `NAN`, `NEG_INF`, `NEG_ZERO`, `POS_INF`, `UNDETERMINED`, `ZERO`
- `ct.RoundingMode`: `APPROX`, `FULL`, `RM`, `RN`, `RP`, `RZ`, `RZI`
- `ct.MemoryOrder`: `ACQUIRE`, `ACQ_REL`, `RELAXED`, `RELEASE`, `WEAK`
- `ct.MemoryScope`: `BLOCK`, `DEVICE`, `NONE`, `SYS`
- `ct.Tile` attributes/methods: `astype`, `dtype`, `extract`, `item`, `ndim`, `permute`, `reshape`, `shape`, `transpose`
- `ct.Array` attributes/methods: `dtype`, `get_raw_memory`, `ndim`, `shape`, `slice`, `strides`, `tiled_view`
- `ct.TiledView` attributes/methods: `atomic_store_add`, `atomic_store_and`, `atomic_store_max`, `atomic_store_min`, `atomic_store_or`, `atomic_store_xor`, `dtype`, `load`, `num_tiles`, `store`, `tile_shape`, `traversal_steps`
- `ct.Slice` attributes/methods: 

## Other public names

`ct.ArrayAnnotation`, `ct.ByTarget`, `ct.Constant`, `ct.ConstantAnnotation`, `ct.DType`, `ct.IndexedWithInt64`, `ct.ListAnnotation`, `ct.Scalar`, `ct.ScalarInt64`, `ct.TileCompilerExecutionError`, `ct.TileCompilerTimeoutError`, `ct.TileError`, `ct.TileInternalError`, `ct.TileRecursionError`, `ct.TileStaticAssertionError`, `ct.TileStaticEvalError`, `ct.TileSyntaxError`, `ct.TileTypeError`, `ct.TileUnsupportedFeatureError`, `ct.TileValueError`, `ct.compiler_timeout`
