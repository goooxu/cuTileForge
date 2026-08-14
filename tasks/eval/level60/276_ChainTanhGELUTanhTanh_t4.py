import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhGELUTanhTanh (tier 4, conv)"""

    def __init__(self, in_channels: int, out_channels: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                              bias=False)

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.gelu((torch.tanh(self.conv(x))), approximate='tanh'))


batch_size = 6
in_channels = 16
out_channels = 32
height = 24
width = 48
def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]
