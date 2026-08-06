import torch
import torch.nn as nn


class Model(nn.Module):
    """MaxPool1d (tier 1, pool)"""

    def __init__(self, kernel_size: int, stride: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


batch_size = 2
channels = 8
length = 256
kernel_size = 2
stride = 2

def get_inputs():
    return [torch.rand(batch_size, channels, length)]


def get_init_inputs():
    return [kernel_size, stride]
