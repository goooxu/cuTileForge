import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLeakyReLUReLU6LeakyReLU (tier 4, matmul)"""

    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.b = nn.Parameter(torch.randn(out_features) * 0.05)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.relu6((torch.nn.functional.leaky_relu(torch.addmm(self.b, x, self.w.t()), negative_slope=0.02))), negative_slope=0.02)


batch_size = 96
in_features = 256
out_features = 256

def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
