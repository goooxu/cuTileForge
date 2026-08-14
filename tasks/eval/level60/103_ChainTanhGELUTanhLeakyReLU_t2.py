import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhGELUTanhLeakyReLU (tier 2, pool)"""

    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.gelu((torch.tanh(self.pool(x))), approximate='tanh'), negative_slope=0.02)


batch_size = 3
channels = 8
height = 24
width = 48
kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size]
