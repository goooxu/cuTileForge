import torch
import torch.nn as nn


class Model(nn.Module):
    """Mish (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish((self.pool(x) ** 2))


batch_size = 25
channels = 64
height = 769
width = 769
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
