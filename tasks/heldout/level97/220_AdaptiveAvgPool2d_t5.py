import torch
import torch.nn as nn


class Model(nn.Module):
    """AdaptiveAvgPool2d (tier 5, pool)"""

    def __init__(self, output_size: int):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


batch_size = 16
channels = 64
height = 512
width = 512
output_size = 2

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [output_size]
