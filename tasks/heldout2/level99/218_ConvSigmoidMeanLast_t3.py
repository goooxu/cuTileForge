import torch
import torch.nn as nn


class Model(nn.Module):
    """ConvSigmoidMeanLast (tier 3, conv)"""

    def __init__(self, in_channels: int, out_channels: int):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                              bias=False)

    def forward(self, x: torch.Tensor):
        return torch.mean(torch.sigmoid(self.conv(x)), dim=-1)


batch_size = 4
in_channels = 16
out_channels = 16
height = 16
width = 16

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]
