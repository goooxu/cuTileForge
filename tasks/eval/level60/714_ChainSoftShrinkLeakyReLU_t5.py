import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftShrinkLeakyReLU (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.softshrink(x, lambd=0.3), negative_slope=0.02)


batch_size = 768
channels = 32
height = 33
width = 33
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
