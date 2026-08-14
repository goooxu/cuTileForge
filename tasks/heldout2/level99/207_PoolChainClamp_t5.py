import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainClamp (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.clamp(self.pool(x), -2.0, 2.0)


batch_size = 16
channels = 64
height = 512
width = 512
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
