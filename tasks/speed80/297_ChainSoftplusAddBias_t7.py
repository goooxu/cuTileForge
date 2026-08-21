import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftplusAddBias (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(x) + 1.5


batch_size = 8888
dim = 20000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
