import torch
import torch.nn as nn


class Model(nn.Module):
    """Sigmoid (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(x)


batch_size = 769
channels = 32
height = 13
width = 13
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
