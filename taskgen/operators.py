"""Operator definitions for synthetic cuTile training tasks.

Each entry describes how to emit a KernelBench-format problem: a PyTorch Model
whose forward is the operator, plus get_inputs/get_init_inputs. These are task
*definitions*, not solutions -- nothing here tells the model how to write the
cuTile kernel.

Operator weighting follows the baseline evaluation's category breakdown. The
model passed 53.8% of activation samples but 2.8% of convolution and 0% of
normalisation and pooling, so those three families are what training data needs
to cover.
"""

from dataclasses import dataclass, field


@dataclass
class Spec:
    """One generated problem, ready to be rendered to a file."""
    name: str
    category: str
    tier: int
    init_sig: str          # __init__ parameters after self
    init_body: str         # __init__ body lines (after super().__init__())
    forward_sig: str       # forward parameters after self
    forward_body: str      # forward body, must end in a return
    consts: dict           # module-level constants
    inputs: str            # body of get_inputs(), must return a list
    init_inputs: str       # body of get_init_inputs(), must return a list
    extra_imports: list = field(default_factory=list)


# --- shape ladders -----------------------------------------------------------
# Tier controls scale, not which operator. The bet behind the curriculum is that
# an operator the model cannot write at 112x64x512x512 may be writable at
# 2x4x16x16, giving rejection sampling a seed to bootstrap from.

NCHW_BY_TIER = {
    2: [(2, 4, 16, 16), (1, 8, 32, 32), (2, 8, 16, 32), (4, 4, 32, 16)],
    3: [(8, 16, 64, 64), (4, 32, 128, 64), (16, 16, 32, 128)],
    5: [(32, 64, 256, 256), (16, 64, 512, 512), (64, 32, 128, 128)],
}

MAT2D_BY_TIER = {
    0: [(256, 512), (128, 1024), (512, 256)],
    1: [(1024, 2048), (2048, 1024), (512, 4096)],
    2: [(64, 128), (32, 256), (128, 64)],
    3: [(2048, 4096), (4096, 2048)],
    5: [(8192, 8192), (4096, 16384)],
}


def shapes_for(table: dict, tier: int):
    """Shape list for a tier, falling back to the nearest lower one.

    The ladders only define the tiers where a distinct scale matters; fusion
    tiers reuse the scale of the tier below rather than needing their own entry.
    """
    for t in range(tier, -1, -1):
        if t in table:
            return table[t]
    return table[min(table)]


# --- normalisation (0/10 in the baseline) ------------------------------------

def layernorm(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    return Spec(
        name="LayerNorm",
        category="norm", tier=tier,
        init_sig="normalized_shape: tuple",
        init_body="        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.ln(x)",
        consts={"batch_size": n, "features": c, "dim1": h, "dim2": w},
        inputs="    return [torch.rand(batch_size, features, dim1, dim2)]",
        init_inputs="    return [(features, dim1, dim2)]",
    )


def rmsnorm(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    return Spec(
        name="RMSNorm",
        category="norm", tier=tier,
        init_sig="num_features: int, eps: float = 1e-5",
        init_body="        self.num_features = num_features\n        self.eps = eps",
        forward_sig="x: torch.Tensor",
        forward_body=("        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)\n"
                      "        return x / rms"),
        consts={"batch_size": n, "features": c, "dim1": h, "dim2": w},
        inputs="    return [torch.rand(batch_size, features, dim1, dim2)]",
        init_inputs="    return [features]",
    )


def groupnorm(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    groups = rng.choice([g for g in (1, 2, 4) if c % g == 0])
    return Spec(
        name="GroupNorm",
        category="norm", tier=tier,
        init_sig="num_features: int, num_groups: int",
        init_body="        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.gn(x)",
        consts={"batch_size": n, "features": c, "dim1": h, "dim2": w,
                "num_groups": groups},
        inputs="    return [torch.rand(batch_size, features, dim1, dim2)]",
        init_inputs="    return [features, num_groups]",
    )


def softmax_dim(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    return Spec(
        name="Softmax",
        category="norm", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return torch.softmax(x, dim=1)",
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


def l2norm(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    return Spec(
        name="L2Norm",
        category="norm", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return x / torch.norm(x, p=2, dim=1, keepdim=True)",
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


# --- pooling (0/6 in the baseline) -------------------------------------------

def maxpool2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    if tier <= 2:
        k, s, p = 2, 2, 0
    else:
        k = rng.choice([2, 3, 4])
        s = rng.choice([1, 2])
        p = rng.choice([0, 1])
    return Spec(
        name="MaxPool2d",
        category="pool", tier=tier,
        init_sig="kernel_size: int, stride: int, padding: int",
        init_body="        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "height": h, "width": w,
                "kernel_size": k, "stride": s, "padding": p},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return [kernel_size, stride, padding]",
    )


def avgpool2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    if tier <= 2:
        k, s, p = 2, 2, 0
    else:
        k = rng.choice([2, 3, 4])
        s = rng.choice([1, 2])
        p = 0
    return Spec(
        name="AvgPool2d",
        category="pool", tier=tier,
        init_sig="kernel_size: int, stride: int, padding: int",
        init_body="        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "height": h, "width": w,
                "kernel_size": k, "stride": s, "padding": p},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return [kernel_size, stride, padding]",
    )


def globalavgpool(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    return Spec(
        name="GlobalAvgPool",
        category="pool", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return torch.mean(x, dim=[2, 3])",
        consts={"batch_size": n, "channels": c, "height": h, "width": w},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return []",
    )


# --- convolution (2.8% in the baseline) --------------------------------------

def conv2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8, 16]) if tier <= 2 else rng.choice([16, 32, 64])
    if tier <= 2:
        k, s, p = 1, 1, 0
    elif tier == 3:
        k, s, p = rng.choice([1, 3]), 1, rng.choice([0, 1])
    else:
        k, s, p = 3, rng.choice([1, 2]), 1
    return Spec(
        name="Conv2d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int",
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,\n"
                   "                              stride=stride, padding=padding, bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kernel_size": k, "stride": s, "padding": p},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels, kernel_size, stride, padding]",
    )


def conv1x1(tier, rng):
    """Pointwise convolution: a matmul in disguise, so a natural stepping stone."""
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8, 16]) if tier <= 2 else rng.choice([32, 64])
    return Spec(
        name="PointwiseConv",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int",
        init_body="        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels]",
    )


# --- matmul and elementwise (the model is already decent here) ---------------

def matmul(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    n = rng.choice([64, 128, 256]) if tier <= 2 else rng.choice([512, 1024, 2048])
    return Spec(
        name="Matmul",
        category="matmul", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="A: torch.Tensor, B: torch.Tensor",
        forward_body="        return torch.matmul(A, B)",
        consts={"M": m, "K": k, "N": n},
        inputs="    return [torch.rand(M, K), torch.rand(K, N)]",
        init_inputs="    return []",
    )


def bmm(tier, rng):
    b = rng.choice([2, 4]) if tier <= 2 else rng.choice([16, 64])
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    n = rng.choice([64, 128]) if tier <= 2 else rng.choice([512, 1024])
    return Spec(
        name="BatchedMatmul",
        category="matmul", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="A: torch.Tensor, B: torch.Tensor",
        forward_body="        return torch.bmm(A, B)",
        consts={"batch_size": b, "M": m, "K": k, "N": n},
        inputs="    return [torch.rand(batch_size, M, K), torch.rand(batch_size, K, N)]",
        init_inputs="    return []",
    )


ELEMENTWISE_OPS = [
    ("ReLU", "torch.relu(x)"),
    ("GELU", "torch.nn.functional.gelu(x)"),
    ("Sigmoid", "torch.sigmoid(x)"),
    ("Tanh", "torch.tanh(x)"),
    ("Swish", "x * torch.sigmoid(x)"),
    ("Softplus", "torch.nn.functional.softplus(x)"),
    ("Square", "x * x"),
    ("Abs", "torch.abs(x)"),
]


def elementwise(tier, rng):
    label, expr = rng.choice(ELEMENTWISE_OPS)
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    return Spec(
        name=label,
        category="elementwise", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


REDUCTION_OPS = [
    ("RowSum", "torch.sum(x, dim=1)"),
    ("RowMax", "torch.max(x, dim=1)[0]"),
    ("RowMean", "torch.mean(x, dim=1)"),
    ("RowMin", "torch.min(x, dim=1)[0]"),
]


def reduction(tier, rng):
    label, expr = rng.choice(REDUCTION_OPS)
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    return Spec(
        name=label,
        category="reduction", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


# --- fusion chains (tier 4) ---------------------------------------------------

FUSION_TAILS = [
    ("ReLU", "torch.relu({})"),
    ("Sigmoid", "torch.sigmoid({})"),
    ("Tanh", "torch.tanh({})"),
    ("Scale", "{} * 2.0"),
    ("AddBias", "{} + 1.5"),
]


def fused_elementwise_chain(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    n_ops = rng.choice([2, 3])
    picked = [rng.choice(FUSION_TAILS) for _ in range(n_ops)]
    expr = "x"
    for _, tmpl in picked:
        expr = tmpl.format(expr)
    label = "".join(p[0] for p in picked)
    return Spec(
        name="Chain%s" % label,
        category="elementwise", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


def fused_matmul_chain(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    n = rng.choice([128, 256]) if tier <= 2 else rng.choice([512, 1024])
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="Matmul%s" % label,
        category="matmul", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="A: torch.Tensor, B: torch.Tensor",
        forward_body="        return %s" % tmpl.format("torch.matmul(A, B)"),
        consts={"M": m, "K": k, "N": n},
        inputs="    return [torch.rand(M, K), torch.rand(K, N)]",
        init_inputs="    return []",
    )


# --- rank ladder (tier 1) -----------------------------------------------------
# The tier-2 probe showed shrinking shapes is not enough for every family.
# Convolution stayed at 0/110 even at 2x4x16x16 with kernel_size=1, and
# PointwiseConv specifically was 0/24 while plain Matmul was 87.5% -- a 1x1
# convolution is a per-pixel matmul, so the blocker is the 4D layout rather than
# the arithmetic. Likewise LayerNorm, RMSNorm and MaxPool2d were flat zero.
#
# These variants hold the operator fixed and lower the tensor rank instead,
# giving those families a rung the model can actually reach.

def layernorm_1d(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, 2))
    return Spec(
        name="LayerNorm1D",
        category="norm", tier=tier,
        init_sig="dim: int",
        init_body="        self.ln = nn.LayerNorm(dim)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.ln(x)",
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return [dim]",
    )


def rmsnorm_1d(tier, rng):
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, 2))
    return Spec(
        name="RMSNorm1D",
        category="norm", tier=tier,
        init_sig="eps: float = 1e-5",
        init_body="        self.eps = eps",
        forward_sig="x: torch.Tensor",
        forward_body=("        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)\n"
                      "        return x / rms"),
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


def maxpool1d(tier, rng):
    n = rng.choice([2, 4, 8])
    c = rng.choice([4, 8])
    length = rng.choice([64, 128, 256])
    return Spec(
        name="MaxPool1d",
        category="pool", tier=tier,
        init_sig="kernel_size: int, stride: int",
        init_body="        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "length": length,
                "kernel_size": 2, "stride": 2},
        inputs="    return [torch.rand(batch_size, channels, length)]",
        init_inputs="    return [kernel_size, stride]",
    )


def rowwise_maxpool_2d(tier, rng):
    """Non-overlapping max over the last axis of a 2D tensor: pooling, rank 2."""
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, 2))
    window = rng.choice([2, 4])
    k = (k // window) * window
    return Spec(
        name="MaxPoolRows",
        category="pool", tier=tier,
        init_sig="window: int",
        init_body="        self.window = window",
        forward_sig="x: torch.Tensor",
        forward_body=("        b, d = x.shape\n"
                      "        return x.reshape(b, d // self.window, self.window).max(dim=2)[0]"),
        consts={"batch_size": m, "dim": k, "window": window},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return [window]",
    )


def conv1d_small(tier, rng):
    n = rng.choice([2, 4])
    c_in = rng.choice([4, 8])
    c_out = rng.choice([4, 8])
    length = rng.choice([64, 128])
    return Spec(
        name="Conv1d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int",
        init_body=("        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,\n"
                   "                              bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c_in, "out_channels": c_out,
                "length": length, "kernel_size": rng.choice([1, 3])},
        inputs="    return [torch.rand(batch_size, in_channels, length)]",
        init_inputs="    return [in_channels, out_channels, kernel_size]",
    )


def pointwise_conv_as_matmul(tier, rng):
    """1x1 convolution expressed on a rank-3 tensor: the bridge from matmul.

    Same arithmetic as PointwiseConv, which the model failed 0/24, but without
    the 4D layout it could not handle.
    """
    n = rng.choice([2, 4])
    c_in = rng.choice([8, 16])
    c_out = rng.choice([8, 16])
    length = rng.choice([64, 256])
    return Spec(
        name="PointwiseConv1d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int",
        init_body="        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c_in, "out_channels": c_out,
                "length": length},
        inputs="    return [torch.rand(batch_size, in_channels, length)]",
        init_inputs="    return [in_channels, out_channels]",
    )


def channel_mean_3d(tier, rng):
    """Reduction over one axis of a rank-3 tensor: a step toward NCHW handling."""
    n = rng.choice([2, 4])
    c = rng.choice([8, 16])
    length = rng.choice([64, 256])
    axis = rng.choice([1, 2])
    return Spec(
        name="Mean3DAxis%d" % axis,
        category="reduction", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return torch.mean(x, dim=%d)" % axis,
        consts={"batch_size": n, "channels": c, "length": length},
        inputs="    return [torch.rand(batch_size, channels, length)]",
        init_inputs="    return []",
    )


# Weighting reflects where the baseline is weak: conv, norm and pool account for
# 92 of the 200 benchmark problems but only 13 were ever solved. Tier 1 is the
# rank ladder for the families that stayed at zero even when shrunk.
BUILDERS = [
    (layernorm_1d,           8, [1]),
    (rmsnorm_1d,             8, [1]),
    (maxpool1d,              8, [1]),
    (rowwise_maxpool_2d,     7, [1]),
    (conv1d_small,           9, [1]),
    (pointwise_conv_as_matmul, 9, [1]),
    (channel_mean_3d,        6, [1]),
    # (builder, weight, tiers it makes sense at)
    (layernorm,              6, [2, 3, 5]),
    (rmsnorm,                6, [2, 3, 5]),
    (groupnorm,              5, [2, 3, 5]),
    (softmax_dim,            6, [0, 2, 3, 5]),
    (l2norm,                 4, [0, 2, 3, 5]),
    (maxpool2d,              7, [2, 3, 5]),
    (avgpool2d,              6, [2, 3, 5]),
    (globalavgpool,          4, [2, 3, 5]),
    (conv2d,                 8, [2, 3, 5]),
    (conv1x1,                6, [2, 3, 5]),
    (matmul,                 4, [0, 2, 3, 5]),
    (bmm,                    4, [2, 3, 5]),
    (elementwise,            3, [0, 2, 3, 5]),
    (reduction,              4, [0, 2, 3, 5]),
    (fused_elementwise_chain, 3, [4]),
    (fused_matmul_chain,      3, [4]),
]
