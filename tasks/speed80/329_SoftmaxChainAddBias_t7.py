import torch
import torch.nn as nn


class Model(nn.Module):
    """SoftmaxChainAddBias (tier 7, norm)"""

    def __init__(self, scale: float):
        super(Model, self).__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return torch.softmax(x * self.scale, dim=1) + 1.5


batch_size = 16384
dim = 8192
scale = 0.125

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [scale]
