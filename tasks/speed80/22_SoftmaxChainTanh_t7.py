import torch
import torch.nn as nn


class Model(nn.Module):
    """SoftmaxChainTanh (tier 7, norm)"""

    def __init__(self, scale: float):
        super(Model, self).__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.softmax(x * self.scale, dim=1))


batch_size = 2048
dim = 65536
scale = 0.125

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [scale]
