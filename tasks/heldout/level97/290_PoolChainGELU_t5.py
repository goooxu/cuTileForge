import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainGELU (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(self.pool(x))


batch_size = 64
channels = 32
height = 128
width = 128
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
