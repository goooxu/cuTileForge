import torch
import torch.nn as nn


class Model(nn.Module):
    """Tanh (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(x)


batch_size = 16
channels = 2
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
