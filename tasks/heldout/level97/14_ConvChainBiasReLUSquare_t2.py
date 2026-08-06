import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvChainBiasReLUSquare (tier 2, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return (torch.relu((self.conv(x) + 0.3)) ** 2)


batch_size = 1
in_channels = 8
out_channels = 8
height = 32
width = 32
kernel_size = 1
padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
