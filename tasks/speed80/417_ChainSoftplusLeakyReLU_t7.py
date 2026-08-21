import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftplusLeakyReLU (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.softplus(x), negative_slope=0.01)


batch_size = 8192
dim = 24576

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
