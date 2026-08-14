import torch
import torch.nn as nn


class Model(nn.Module):
    """HardShrink (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardshrink(x, lambd=0.5)


batch_size = 512
channels = 32
height = 22
width = 22

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
