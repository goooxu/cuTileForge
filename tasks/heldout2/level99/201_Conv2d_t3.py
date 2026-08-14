import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d (tier 3, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


batch_size = 4
in_channels = 32
out_channels = 32
height = 128
width = 64
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
