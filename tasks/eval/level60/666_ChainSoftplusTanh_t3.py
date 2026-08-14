import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftplusTanh (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.softplus(x))


batch_size = 1536
channels = 32
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
