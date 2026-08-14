import torch
import torch.nn as nn


class Model(nn.Module):
    """Scale (tier 1, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor):
        return (self.conv(x)) * 2.0


batch_size = 3
in_channels = 4
out_channels = 4
length = 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
