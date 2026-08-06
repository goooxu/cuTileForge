import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainGELUSigmoid (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.nn.functional.gelu(self.pool(x)))


batch_size = 4
channels = 4
height = 32
width = 16
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
