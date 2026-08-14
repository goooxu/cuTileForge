import torch
import torch.nn as nn


class Model(nn.Module):
    """LeakyReLU (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(x, negative_slope=0.02)


batch_size = 48
dim = 384
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
