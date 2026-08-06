import torch
import torch.nn as nn


class Model(nn.Module):
    """MaxPool2d (tier 5, pool)"""

    def __init__(self, kernel_size: int, stride: int, padding: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


batch_size = 32
channels = 64
height = 256
width = 256
kernel_size = 2
stride = 2
padding = 0

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size, stride, padding]
