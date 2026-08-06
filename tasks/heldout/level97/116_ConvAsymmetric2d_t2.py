import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvAsymmetric2d (tier 2, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kh: int, kw: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw),
                              padding=(kh // 2, kw // 2),
                              bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


batch_size = 1
in_channels = 8
out_channels = 4
height = 32
width = 32
kh = 1
kw = 5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kh, kw]
