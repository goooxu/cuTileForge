import torch
import torch.nn as nn


class Model(nn.Module):
    """HardShrink (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardshrink(x, lambd=0.5)


batch_size = 64
dim = 128

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
