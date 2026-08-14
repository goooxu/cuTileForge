import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLeakyReLUGELU (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.nn.functional.leaky_relu(x, negative_slope=0.02))


batch_size = 768
channels = 32
height = 12
width = 12
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
