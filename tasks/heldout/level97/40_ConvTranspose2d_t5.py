import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvTranspose2d (tier 5, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride=stride,
                                       bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


batch_size = 64
in_channels = 32
out_channels = 16
height = 128
width = 128
kernel_size = 3
stride = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride]
