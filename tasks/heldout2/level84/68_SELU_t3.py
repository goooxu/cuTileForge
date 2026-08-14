import torch
import torch.nn as nn


class Model(nn.Module):
    """SELU (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.selu(x)


batch_size = 512
channels = 32
height = 8
width = 8

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
