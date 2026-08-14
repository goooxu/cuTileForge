import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhAddBias (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return ((torch.tanh(self.pool(x)) + 0.3)) + 1.5


batch_size = 2
channels = 8
height = 48
width = 48
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
