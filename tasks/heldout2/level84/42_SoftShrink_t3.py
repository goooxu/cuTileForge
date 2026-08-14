import torch
import torch.nn as nn


class Model(nn.Module):
    """SoftShrink (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softshrink(x, lambd=0.5)


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
