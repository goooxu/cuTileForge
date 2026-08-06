import torch
import torch.nn as nn


class Model(nn.Module):
    """AvgPool2d (tier 3, pool)"""

    def __init__(self, kernel_size: int, stride: int, padding: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


batch_size = 4
channels = 32
height = 128
width = 64
kernel_size = 3
stride = 1
padding = 0

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size, stride, padding]
