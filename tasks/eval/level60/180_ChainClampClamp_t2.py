import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampClamp (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.clamp(self.pool(x), -2.0, 2.0), min=-1.0, max=1.0)


batch_size = 6
channels = 4
height = 48
width = 24
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
