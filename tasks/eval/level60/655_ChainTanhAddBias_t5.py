import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhAddBias (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.tanh(x)) + 1.5


batch_size = 1536
channels = 32
height = 24
width = 24
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
