import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELULeakyReLUCELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.celu((torch.nn.functional.leaky_relu((torch.nn.functional.gelu(x)), negative_slope=0.02)), alpha=1.25)


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
