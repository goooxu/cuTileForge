import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampLeakyReLU (tier 4, matmul)"""

    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.b = nn.Parameter(torch.randn(out_features) * 0.05)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.clamp(torch.addmm(self.b, x, self.w.t()), min=-1.0, max=1.0), negative_slope=0.02)


batch_size = 192
in_features = 128
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
