import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampELU (tier 5, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.clamp(self.pool(x), -2.0, 2.0))


batch_size = 97
channels = 32
height = 193
width = 193
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
