import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLUMish (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish((torch.relu(x)))


batch_size = 33311
dim = 33313
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
