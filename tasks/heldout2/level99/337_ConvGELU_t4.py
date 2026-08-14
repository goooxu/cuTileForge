import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvGELU (tier 4, conv)"""

    def __init__(self, in_channels: int, out_channels: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                              bias=False)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(self.conv(x))


batch_size = 8
in_channels = 16
out_channels = 16
height = 32
width = 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]
