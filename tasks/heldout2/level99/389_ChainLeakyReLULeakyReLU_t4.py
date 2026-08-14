import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLeakyReLULeakyReLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.leaky_relu(x, negative_slope=0.01), negative_slope=0.01)


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
