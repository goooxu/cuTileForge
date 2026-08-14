import torch
import torch.nn as nn


class Model(nn.Module):
    """Clamp (tier 5, pool)"""

    def __init__(self, kernel_size: int, stride: int, padding: int):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        return torch.clamp(self.pool(x), min=-1.0, max=1.0)


batch_size = 24
channels = 64
height = 768
width = 768
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return [kernel_size, stride, padding]
