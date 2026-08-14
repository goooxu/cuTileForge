import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLUTanhShrinkClamp (tier 3, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.tanhshrink((torch.relu(self.pool(x)))), min=-1.0, max=1.0)


batch_size = 24
channels = 16
height = 48
width = 192
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
