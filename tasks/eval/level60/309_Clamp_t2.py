import torch
import torch.nn as nn


class Model(nn.Module):
    """Clamp (tier 2, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return torch.clamp((self.conv(x) * 1.7), -2.0, 2.0)


batch_size = 3
in_channels = 8
out_channels = 4
height = 24
width = 48
kernel_size = 1
padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
