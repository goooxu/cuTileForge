import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhGELUTanhSigmoid (tier 5, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.nn.functional.gelu(((torch.tanh(self.conv(x)) * 1.7)), approximate='tanh'))


batch_size = 96
in_channels = 32
out_channels = 32
height = 192
width = 192
kernel_size = 3
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
