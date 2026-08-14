import torch
import torch.nn as nn


class Model(nn.Module):
    """HardTanh (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)


batch_size = 256
channels = 32
height = 11
width = 11

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
