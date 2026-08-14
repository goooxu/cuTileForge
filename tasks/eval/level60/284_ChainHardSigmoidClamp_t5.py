import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSigmoidClamp (tier 5, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.hardsigmoid(((self.conv(x) * 1.7))), min=-1.0, max=1.0)


batch_size = 24
in_channels = 64
out_channels = 16
height = 768
width = 768
kernel_size = 3
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]
