import torch
import torch.nn as nn


class Model(nn.Module):
    """GELUTanh (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(x, approximate='tanh')


batch_size = 512
channels = 64
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
