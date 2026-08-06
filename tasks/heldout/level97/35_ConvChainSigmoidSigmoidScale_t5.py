import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvChainSigmoidSigmoidScale (tier 5, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return (torch.sigmoid(torch.sigmoid(self.conv(x))) * 1.7)


batch_size = 64
in_channels = 32
out_channels = 32
height = 128
width = 128
kernel_size = 3
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
