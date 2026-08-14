import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainClamp (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


batch_size = 24
channels = 64
height = 768
width = 768
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
