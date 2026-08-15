import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftplusAddBias (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.softplus(x)) + 1.5


batch_size = 2304
dim = 524288
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
