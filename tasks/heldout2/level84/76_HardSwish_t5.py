import torch
import torch.nn as nn


class Model(nn.Module):
    """HardSwish (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(x)


batch_size = 2048
channels = 128
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
