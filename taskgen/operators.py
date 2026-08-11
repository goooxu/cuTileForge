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

# Rank-3 and rank-5 ladders, for the 1D and 3D halves of the API surface. The
# rank matters as much as the size: grid and tile rank have to match the array's,
# and rank_mismatch was the single most common failure at baseline.
NCL_BY_TIER = {
    1: [(2, 4, 32), (1, 8, 64)],
    2: [(2, 4, 64), (4, 8, 128), (2, 16, 64)],
    3: [(8, 32, 512), (16, 64, 256)],
    5: [(32, 64, 4096), (64, 128, 1024)],
}

NCDHW_BY_TIER = {
    2: [(1, 4, 4, 8, 8), (2, 4, 2, 8, 8)],
    3: [(2, 8, 8, 16, 16), (4, 16, 4, 32, 32)],
    5: [(8, 32, 16, 64, 64), (4, 64, 8, 128, 128)],
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


# The pointwise activation surface, enumerated rather than sampled. This family
# is 29 of the benchmark's 200 problems and had almost no training material: the
# elementwise builder above covers eight ops and only in two dimensions, so the
# frontier carried 3.4% elementwise against a 14.5% dev share, and stacking RL on
# top of the distilled model lost two activation problems rather than gaining any.
ACTIVATION_OPS = [
    ("ReLU", "torch.relu(x)"),
    ("LeakyReLU", "torch.nn.functional.leaky_relu(x, negative_slope=0.01)"),
    ("Sigmoid", "torch.sigmoid(x)"),
    ("Tanh", "torch.tanh(x)"),
    ("GELU", "torch.nn.functional.gelu(x)"),
    ("GELUTanh", "torch.nn.functional.gelu(x, approximate='tanh')"),
    ("SELU", "torch.selu(x)"),
    ("ELU", "torch.nn.functional.elu(x, alpha=1.0)"),
    ("CELU", "torch.nn.functional.celu(x, alpha=1.0)"),
    ("Softplus", "torch.nn.functional.softplus(x)"),
    ("Softsign", "torch.nn.functional.softsign(x)"),
    ("HardSigmoid", "torch.nn.functional.hardsigmoid(x)"),
    ("HardSwish", "torch.nn.functional.hardswish(x)"),
    ("HardTanh", "torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)"),
    ("HardShrink", "torch.nn.functional.hardshrink(x, lambd=0.5)"),
    ("SoftShrink", "torch.nn.functional.softshrink(x, lambd=0.5)"),
    ("TanhShrink", "torch.nn.functional.tanhshrink(x)"),
    ("LogSigmoid", "torch.nn.functional.logsigmoid(x)"),
    ("SiLU", "torch.nn.functional.silu(x)"),
    ("Mish", "torch.nn.functional.mish(x)"),
    ("ReLU6", "torch.nn.functional.relu6(x)"),
    ("Softmax", "torch.softmax(x, dim=-1)"),
    ("LogSoftmax", "torch.log_softmax(x, dim=-1)"),
    ("Softmin", "torch.nn.functional.softmin(x, dim=-1)"),
]


def activation(tier, rng):
    """One pointwise activation, at 2D, 3D or 4D.

    Rank is varied because the benchmark's activation problems are not all
    matrices, and a kernel written only for 2D does not carry over: the tile
    indexing changes with rank.
    """
    label, expr = rng.choice(ACTIVATION_OPS)
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    rank = rng.choice([2, 2, 3, 4]) if tier >= 2 else 2
    if rank == 2:
        consts = {"batch_size": m, "dim": k}
        inputs = "    return [torch.randn(batch_size, dim)]"
    elif rank == 3:
        c = max(2, k // 64)
        consts = {"batch_size": max(1, m // 4), "channels": c,
                  "length": max(8, k // max(c, 1))}
        inputs = "    return [torch.randn(batch_size, channels, length)]"
    else:
        c = max(2, min(32, k // 32))
        side = max(4, int((k // max(c, 1)) ** 0.5))
        consts = {"batch_size": max(1, m // 8), "channels": c,
                  "height": side, "width": side}
        inputs = "    return [torch.randn(batch_size, channels, height, width)]"
    return Spec(
        name=label,
        category="activation", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts=consts,
        inputs=inputs,
        init_inputs="    return []",
    )


# Losses that take float tensors on both sides. Integer class targets are left
# out on purpose: the evaluation protocol forces fp32 on every input, which is
# what made the benchmark's Level 4 unusable here, and a cast index tensor is
# silently wrong rather than loudly broken.
LOSS_OPS = [
    ("MSELoss", "torch.mean((predictions - targets) ** 2)"),
    ("L1Loss", "torch.mean(torch.abs(predictions - targets))"),
    ("HuberLoss", "torch.nn.functional.huber_loss(predictions, targets)"),
    ("SmoothL1Loss", "torch.nn.functional.smooth_l1_loss(predictions, targets)"),
    ("HingeLoss", "torch.mean(torch.clamp(1 - predictions * targets, min=0))"),
    ("KLDivLoss",
     "torch.nn.functional.kl_div(torch.log_softmax(predictions, dim=-1), "
     "torch.softmax(targets, dim=-1), reduction='batchmean')"),
    ("CosineSimilarityLoss",
     "torch.mean(1 - torch.nn.functional.cosine_similarity("
     "predictions, targets, dim=-1))"),
]


def loss_fn(tier, rng):
    label, expr = rng.choice(LOSS_OPS)
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    return Spec(
        name=label,
        category="loss", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="predictions: torch.Tensor, targets: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.randn(batch_size, dim), "
               "torch.randn(batch_size, dim)]",
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


# ---------------------------------------------------------------------------
# The rest of the torch.nn convolution, pooling and normalisation surface
# ---------------------------------------------------------------------------
# Everything above this point covers Conv2d and 1x1 conv and nothing else, which
# is why 102 of the 200 benchmark problems have never been solved by any version:
# 64 of them are convolutions in forms no training task ever presented --
# transposed, dilated, depthwise, grouped, 3D, asymmetric. Enumerating the API
# surface is mechanical and it is the only way those families get a foothold.
#
# Every builder here spans tiers, because that foothold is what made the
# difference before: rank-3 conv at tier 1 is what took convolution from 5 solved
# problems to 14, not more tasks at full difficulty.


def conv_transpose2d(tier, rng):
    n, c, h, w = shapes_for(NCHW_BY_TIER, tier)[0] if tier <= 2 else \
        rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    k = 2 if tier <= 2 else rng.choice([2, 3, 4])
    s = 2 if tier <= 2 else rng.choice([1, 2])
    return Spec(
        name="ConvTranspose2d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int, stride: int",
        init_body=("        self.conv = nn.ConvTranspose2d(in_channels, out_channels,\n"
                   "                                       kernel_size, stride=stride,\n"
                   "                                       bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kernel_size": k, "stride": s},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels, kernel_size, stride]",
    )


def conv_transpose1d(tier, rng):
    n, c, length = rng.choice(shapes_for(NCL_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    return Spec(
        name="ConvTranspose1d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int, stride: int",
        init_body=("        self.conv = nn.ConvTranspose1d(in_channels, out_channels,\n"
                   "                                       kernel_size, stride=stride,\n"
                   "                                       bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "length": length, "kernel_size": 2, "stride": 2},
        inputs="    return [torch.rand(batch_size, in_channels, length)]",
        init_inputs="    return [in_channels, out_channels, kernel_size, stride]",
    )


def conv_dilated2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    d = rng.choice([2, 3])
    # Padding chosen to keep the output non-degenerate at the smaller tiers.
    return Spec(
        name="ConvDilated2d",
        category="conv", tier=tier,
        init_sig=("in_channels: int, out_channels: int, kernel_size: int, "
                  "dilation: int, padding: int"),
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,\n"
                   "                              dilation=dilation, padding=padding,\n"
                   "                              bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kernel_size": 3, "dilation": d,
                "padding": d},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs=("    return [in_channels, out_channels, kernel_size, dilation, "
                     "padding]"),
    )


def conv_depthwise2d(tier, rng):
    """groups == in_channels: each channel gets its own filter, no mixing."""
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    k = 3 if tier > 2 else 1
    return Spec(
        name="ConvDepthwise2d",
        category="conv", tier=tier,
        init_sig="channels: int, kernel_size: int, padding: int",
        init_body=("        self.conv = nn.Conv2d(channels, channels, kernel_size,\n"
                   "                              padding=padding, groups=channels,\n"
                   "                              bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "channels": c, "height": h, "width": w,
                "kernel_size": k, "padding": k // 2},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return [channels, kernel_size, padding]",
    )


def conv_grouped2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    groups = rng.choice([g for g in (2, 4) if c % g == 0] or [1])
    out_c = c if c % groups == 0 else c
    k = 3 if tier > 2 else 1
    return Spec(
        name="ConvGrouped2d",
        category="conv", tier=tier,
        init_sig=("in_channels: int, out_channels: int, kernel_size: int, "
                  "groups: int, padding: int"),
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,\n"
                   "                              groups=groups, padding=padding,\n"
                   "                              bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kernel_size": k, "groups": groups,
                "padding": k // 2},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs=("    return [in_channels, out_channels, kernel_size, groups, "
                     "padding]"),
    )


def conv_asymmetric2d(tier, rng):
    """Rectangular kernel and stride: the indexing the model most often gets wrong."""
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    kh, kw = rng.choice([(1, 3), (3, 1), (1, 5), (5, 1)])
    return Spec(
        name="ConvAsymmetric2d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kh: int, kw: int",
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw),\n"
                   "                              padding=(kh // 2, kw // 2),\n"
                   "                              bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kh": kh, "kw": kw},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels, kh, kw]",
    )


def conv3d_small(tier, rng):
    n, c, d, h, w = rng.choice(shapes_for(NCDHW_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    return Spec(
        name="Conv3d",
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int",
        init_body=("        self.conv = nn.Conv3d(in_channels, out_channels,\n"
                   "                              kernel_size, bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.conv(x)",
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "depth": d, "height": h, "width": w,
                "kernel_size": 1 if tier <= 2 else 3},
        inputs="    return [torch.rand(batch_size, in_channels, depth, height, width)]",
        init_inputs="    return [in_channels, out_channels, kernel_size]",
    )


def batchnorm2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    return Spec(
        name="BatchNorm2d",
        category="norm", tier=tier,
        # eval() so the running statistics are fixed and the forward is
        # deterministic; in train mode the reference would update state between
        # the two models' calls and never match.
        init_sig="num_features: int",
        init_body=("        self.bn = nn.BatchNorm2d(num_features)\n"
                   "        self.bn.eval()"),
        forward_sig="x: torch.Tensor",
        forward_body="        return self.bn(x)",
        consts={"batch_size": n, "num_features": c, "height": h, "width": w},
        inputs="    return [torch.rand(batch_size, num_features, height, width)]",
        init_inputs="    return [num_features]",
    )


def instancenorm2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    return Spec(
        name="InstanceNorm2d",
        category="norm", tier=tier,
        init_sig="num_features: int",
        init_body="        self.inorm = nn.InstanceNorm2d(num_features)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.inorm(x)",
        consts={"batch_size": n, "num_features": c, "height": h, "width": w},
        inputs="    return [torch.rand(batch_size, num_features, height, width)]",
        init_inputs="    return [num_features]",
    )


def maxpool3d(tier, rng):
    n, c, d, h, w = rng.choice(shapes_for(NCDHW_BY_TIER, tier))
    return Spec(
        name="MaxPool3d",
        category="pool", tier=tier,
        init_sig="kernel_size: int",
        init_body="        self.pool = nn.MaxPool3d(kernel_size)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "depth": d, "height": h,
                "width": w, "kernel_size": 2},
        inputs="    return [torch.rand(batch_size, channels, depth, height, width)]",
        init_inputs="    return [kernel_size]",
    )


def adaptive_avgpool2d(tier, rng):
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out = rng.choice([1, 2]) if tier <= 2 else rng.choice([1, 2, 4])
    return Spec(
        name="AdaptiveAvgPool2d",
        category="pool", tier=tier,
        init_sig="output_size: int",
        init_body="        self.pool = nn.AdaptiveAvgPool2d(output_size)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "height": h, "width": w,
                "output_size": out},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return [output_size]",
    )


def avgpool1d(tier, rng):
    n, c, length = rng.choice(shapes_for(NCL_BY_TIER, tier))
    return Spec(
        name="AvgPool1d",
        category="pool", tier=tier,
        init_sig="kernel_size: int",
        init_body="        self.pool = nn.AvgPool1d(kernel_size)",
        forward_sig="x: torch.Tensor",
        forward_body="        return self.pool(x)",
        consts={"batch_size": n, "channels": c, "length": length,
                "kernel_size": 2},
        inputs="    return [torch.rand(batch_size, channels, length)]",
        init_inputs="    return [kernel_size]",
    )


# ---------------------------------------------------------------------------
# Fusion tasks, for teaching speed rather than correctness
# ---------------------------------------------------------------------------
# Everything above exists to teach the model to write a *correct* kernel, and
# uses small shapes so verification is cheap. That makes those tasks useless for
# teaching it to write a *fast* one: at tier 2, (2, 4, 16, 16) is 2048 elements,
# the launch dominates, both implementations measure the same, and there is
# nothing in the comparison to learn from.
#
# These shapes are all large enough to be memory-bound, so a fused kernel can
# actually win. The advantage a kernel DSL has over a library is not doing any
# single operator faster -- torch calls cuBLAS and cuDNN -- but not writing the
# intermediate out between operators, so the payoff scales with how big the
# intermediates are and how many of them a chain has.
PERF_2D = [(4096, 4096), (8192, 2048), (2048, 8192), (8192, 8192)]
PERF_NCHW = [(16, 64, 128, 128), (8, 128, 128, 128), (32, 32, 256, 256)]


def long_elementwise_chain(tier, rng):
    """A long elementwise chain on a large tensor.

    The easiest fusion win there is, and one the model already gets: its best
    kernel on the whole benchmark is a 4.95x on Level 1's GELU, which is exactly
    this shape of problem at 8192x8192. Included as the curriculum's floor and as
    a control -- if speed does not improve here, it will not improve anywhere.
    """
    m, k = rng.choice(PERF_2D)
    picked = [rng.choice(FUSION_TAILS) for _ in range(rng.choice([4, 6, 8]))]
    expr = "x"
    for _, tmpl in picked:
        expr = tmpl.format(expr)
    return Spec(
        name="LongChain%d" % len(picked),
        category="elementwise", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


def fused_matmul_bias_act(tier, rng):
    """Matmul, bias, activation: torch must land the product before the bias."""
    m, k = rng.choice(PERF_2D)
    n = rng.choice([1024, 2048, 4096])
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="MatmulBias%s" % label,
        category="matmul", tier=tier,
        init_sig="n: int",
        init_body="        self.bias = nn.Parameter(torch.randn(n))",
        forward_sig="A: torch.Tensor, B: torch.Tensor",
        forward_body="        return %s" % tmpl.format("torch.matmul(A, B) + self.bias"),
        consts={"M": m, "K": k, "N": n},
        inputs="    return [torch.rand(M, K), torch.rand(K, N)]",
        init_inputs="    return [N]",
    )


def fused_matmul_residual(tier, rng):
    """Matmul, activation, then a residual add: three intermediates to avoid."""
    m, k = rng.choice(PERF_2D)
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="MatmulResidual%s" % label,
        category="matmul", tier=tier,
        init_sig="",
        init_body="        pass",
        forward_sig="A: torch.Tensor, B: torch.Tensor, C: torch.Tensor",
        forward_body="        return %s + C" % tmpl.format("torch.matmul(A, B)"),
        consts={"M": m, "K": k, "N": m},
        inputs="    return [torch.rand(M, K), torch.rand(K, N), torch.rand(M, N)]",
        init_inputs="    return []",
    )


def fused_conv_bias_act(tier, rng):
    """1x1 conv, bias, activation, at a size where the intermediate is large."""
    n, c, h, w = rng.choice(PERF_NCHW)
    out_c = rng.choice([32, 64])
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="ConvBias%s" % label,
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int",
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, 1,\n"
                   "                              bias=False)\n"
                   "        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1))"),
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % tmpl.format("self.conv(x) + self.bias"),
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels]",
    )


def fused_norm_residual(tier, rng):
    """Normalise, add a residual, activate: the classic transformer block tail."""
    m, k = rng.choice(PERF_2D)
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="NormResidual%s" % label,
        category="norm", tier=tier,
        init_sig="dim: int, eps: float = 1e-5",
        init_body="        self.eps = eps",
        forward_sig="x: torch.Tensor, r: torch.Tensor",
        forward_body=(
            "        mean = x.mean(dim=1, keepdim=True)\n"
            "        var = x.var(dim=1, keepdim=True, unbiased=False)\n"
            "        h = (x - mean) / torch.sqrt(var + self.eps)\n"
            "        return %s" % tmpl.format("h + r")),
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]",
        init_inputs="    return [dim]",
    )


def fused_softmax_chain(tier, rng):
    """Scale, softmax, activation: attention's tail, and reduction-bound."""
    m, k = rng.choice(PERF_2D)
    label, tmpl = rng.choice(FUSION_TAILS)
    return Spec(
        name="SoftmaxChain%s" % label,
        category="norm", tier=tier,
        init_sig="scale: float",
        init_body="        self.scale = scale",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % tmpl.format(
            "torch.softmax(x * self.scale, dim=1)"),
        consts={"batch_size": m, "dim": k, "scale": 0.125},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return [scale]",
    )


# ---------------------------------------------------------------------------
# Compositional chains
# ---------------------------------------------------------------------------
# KernelBench Level 2 is entirely operator chains, and it is the half nothing has
# moved: 16 of 100 problems at baseline, 15 after six rounds of training. The
# builders above emit one operator each, so no task has ever had Level 2's shape.
#
# Chains are also where the diversity is. Picking 3 heads from a dozen and 2 tails
# from six gives thousands of distinct combinations from a few dozen lines, which
# no amount of enumerating single operators reaches.

# Chain heads: produce a tensor from the input. Each entry is
# (label, init lines, forward expression, const requirements).
CHAIN_HEADS_2D = [
    ("Matmul", "", "torch.matmul(A, B)"),
    ("Softmax", "", "torch.softmax(x, dim=1)"),
    ("LayerNorm", "        self.ln = nn.LayerNorm(dim)", "self.ln(x)"),
    ("L2Norm", "", "x / torch.norm(x, p=2, dim=1, keepdim=True)"),
    ("RowMean", "", "x.mean(dim=1, keepdim=True)"),
]

# Chain tails: transform a tensor elementwise. Composable in any order and any
# depth, which is what makes the space combinatorial.
CHAIN_TAILS = [
    ("ReLU", "torch.relu({})"),
    ("Sigmoid", "torch.sigmoid({})"),
    ("Tanh", "torch.tanh({})"),
    ("GELU", "torch.nn.functional.gelu({})"),
    ("Scale", "({} * 1.7)"),
    ("Bias", "({} + 0.3)"),
    ("Square", "({} ** 2)"),
    ("Clamp", "torch.clamp({}, -2.0, 2.0)"),
]


def operator_chain(tier, rng):
    """A head plus two to four elementwise tails, the shape Level 2 problems take.

    Large shapes at the upper tiers so the fusion payoff is measurable: the win a
    tile DSL has over a library is not materialising the intermediates, and that
    is proportional to how big they are.
    """
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    n_tails = rng.choice([2, 3, 4])
    tails = [rng.choice(CHAIN_TAILS) for _ in range(n_tails)]
    head_label, head_init, head_expr = rng.choice(CHAIN_HEADS_2D)

    expr = head_expr
    for _, tmpl in tails:
        expr = tmpl.format(expr)
    label = head_label + "".join(t[0] for t in tails)

    if head_label == "Matmul":
        n = rng.choice([64, 128]) if tier <= 2 else rng.choice([512, 1024])
        return Spec(
            name="Chain%s" % label,
            category="matmul", tier=tier,
            init_sig="", init_body="        pass",
            forward_sig="A: torch.Tensor, B: torch.Tensor",
            forward_body="        return %s" % expr,
            consts={"M": m, "K": k, "N": n},
            inputs="    return [torch.rand(M, K), torch.rand(K, N)]",
            init_inputs="    return []",
        )

    needs_dim = head_label == "LayerNorm"
    return Spec(
        name="Chain%s" % label,
        category="norm" if head_label in ("Softmax", "LayerNorm", "L2Norm") else "reduction",
        tier=tier,
        init_sig="dim: int" if needs_dim else "",
        init_body=head_init or "        pass",
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim)]",
        init_inputs="    return [dim]" if needs_dim else "    return []",
    )


def conv_chain(tier, rng):
    """Convolution followed by elementwise work: Level 2's most common shape.

    Half the benchmark is convolution and half of Level 2's chains are anchored on
    one, so this is the intersection of the two weakest areas.
    """
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    out_c = rng.choice([4, 8]) if tier <= 2 else rng.choice([16, 32])
    k = 1 if tier <= 2 else rng.choice([1, 3])
    tails = [rng.choice(CHAIN_TAILS) for _ in range(rng.choice([1, 2, 3]))]
    expr = "self.conv(x)"
    for _, tmpl in tails:
        expr = tmpl.format(expr)
    return Spec(
        name="ConvChain%s" % "".join(t[0] for t in tails),
        category="conv", tier=tier,
        init_sig="in_channels: int, out_channels: int, kernel_size: int, padding: int",
        init_body=("        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,\n"
                   "                              padding=padding, bias=False)"),
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": n, "in_channels": c, "out_channels": out_c,
                "height": h, "width": w, "kernel_size": k, "padding": k // 2},
        inputs="    return [torch.rand(batch_size, in_channels, height, width)]",
        init_inputs="    return [in_channels, out_channels, kernel_size, padding]",
    )


def pool_chain(tier, rng):
    """Pooling plus elementwise work. Pooling is 10 problems and solved 0-3."""
    n, c, h, w = rng.choice(shapes_for(NCHW_BY_TIER, tier))
    tails = [rng.choice(CHAIN_TAILS) for _ in range(rng.choice([1, 2]))]
    expr = "self.pool(x)"
    for _, tmpl in tails:
        expr = tmpl.format(expr)
    which = rng.choice(["MaxPool2d", "AvgPool2d"])
    return Spec(
        name="PoolChain%s" % "".join(t[0] for t in tails),
        category="pool", tier=tier,
        init_sig="kernel_size: int",
        init_body="        self.pool = nn.%s(kernel_size)" % which,
        forward_sig="x: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": n, "channels": c, "height": h, "width": w,
                "kernel_size": 2},
        inputs="    return [torch.rand(batch_size, channels, height, width)]",
        init_inputs="    return [kernel_size]",
    )


def residual_chain(tier, rng):
    """Two inputs combined then transformed: the transformer block tail."""
    m, k = rng.choice(shapes_for(MAT2D_BY_TIER, tier))
    tails = [rng.choice(CHAIN_TAILS) for _ in range(rng.choice([1, 2, 3]))]
    expr = "(x + r)"
    for _, tmpl in tails:
        expr = tmpl.format(expr)
    return Spec(
        name="Residual%s" % "".join(t[0] for t in tails),
        category="elementwise", tier=tier,
        init_sig="", init_body="        pass",
        forward_sig="x: torch.Tensor, r: torch.Tensor",
        forward_body="        return %s" % expr,
        consts={"batch_size": m, "dim": k},
        inputs="    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]",
        init_inputs="    return []",
    )


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
    # Tier 6 is the speed curriculum: large shapes and long chains, the only
    # tier where a measured speedup means anything. Ordered easiest first --
    # the long elementwise chain is where the model already beats torch.
    # The rest of the torch.nn surface. Weighted heavily because these are the
    # forms the 64 unsolved convolution problems actually take, and no task has
    # ever presented them.
    (conv_transpose2d,       9, [2, 3, 5]),
    (conv_transpose1d,       8, [1, 2, 3]),
    (conv_dilated2d,         8, [2, 3, 5]),
    (conv_depthwise2d,       9, [2, 3, 5]),
    (conv_grouped2d,         8, [2, 3, 5]),
    (conv_asymmetric2d,      8, [2, 3, 5]),
    (conv3d_small,           7, [2, 3]),
    (batchnorm2d,            6, [2, 3, 5]),
    (instancenorm2d,         6, [2, 3, 5]),
    (maxpool3d,              5, [2, 3]),
    (adaptive_avgpool2d,     6, [2, 3, 5]),
    (avgpool1d,              5, [1, 2, 3]),
    # Activation and loss. Together these are 35 of the 200 benchmark problems
    # and had no dedicated builder at all until now -- the closest thing was the
    # eight-op elementwise family at 2D. Weighted like the conv surface because
    # the gap is the same kind: a whole torch API face with no task behind it.
    (activation,             9, [0, 1, 2, 3, 5]),
    (loss_fn,                7, [0, 2, 3, 5]),
    # Chains. Level 2 is 100 of the 200 problems, is entirely chains, and has
    # moved 16 -> 15 across six rounds, so this is where the weight goes.
    (operator_chain,        12, [2, 3, 5]),
    (conv_chain,            12, [2, 3, 5]),
    (pool_chain,             8, [2, 3, 5]),
    (residual_chain,         6, [2, 3, 5]),
    (long_elementwise_chain, 10, [6]),
    (fused_softmax_chain,     8, [6]),
    (fused_norm_residual,     8, [6]),
    (fused_matmul_bias_act,   9, [6]),
    (fused_matmul_residual,   8, [6]),
    (fused_conv_bias_act,     9, [6]),
]
