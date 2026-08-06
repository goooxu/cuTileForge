import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvChainReLUSigmoidSigmoid (tier 3, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.sigmoid(torch.relu(self.conv(x))))


batch_size = 16
in_channels = 16
out_channels = 16
height = 32
width = 128
kernel_size = 3
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
