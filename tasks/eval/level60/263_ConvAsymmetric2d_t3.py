import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvAsymmetric2d (tier 3, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kh: int, kw: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw),
                              padding=(kh // 2, kw // 2),
                              bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


batch_size = 24
in_channels = 16
out_channels = 32
height = 48
width = 192
kh = 1
kw = 5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kh, kw]
