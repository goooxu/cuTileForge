import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSigmoidMish (tier 3, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(torch.nn.functional.hardsigmoid(((self.pool(x) * 1.7))))


batch_size = 6
channels = 32
height = 192
width = 96
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
