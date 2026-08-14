import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU (tier 3, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.relu(self.pool(x))


batch_size = 24
channels = 16
height = 48
width = 192
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
