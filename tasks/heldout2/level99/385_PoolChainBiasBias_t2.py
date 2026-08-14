import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainBiasBias (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return ((self.pool(x) + 0.3) + 0.3)


batch_size = 4
channels = 4
height = 32
width = 16
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
