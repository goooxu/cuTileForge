import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainBiasSigmoid (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid((self.pool(x) + 0.3))


batch_size = 16
channels = 64
height = 512
width = 512
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
