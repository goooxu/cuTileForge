import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv1d (tier 1, conv)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


batch_size = 4
in_channels = 4
out_channels = 4
length = 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
