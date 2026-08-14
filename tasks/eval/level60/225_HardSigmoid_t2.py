import torch
import torch.nn as nn


class Model(nn.Module):
    """HardSigmoid (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardsigmoid(x)


batch_size = 12
channels = 4
height = 8
width = 8
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
