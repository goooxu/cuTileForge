import torch
import torch.nn as nn


class Model(nn.Module):
    """PoolChainSigmoidSquare (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return (torch.sigmoid(self.pool(x)) ** 2)


batch_size = 2
channels = 4
height = 16
width = 16
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
