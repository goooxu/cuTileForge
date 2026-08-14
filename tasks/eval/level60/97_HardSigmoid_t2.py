import torch
import torch.nn as nn


class Model(nn.Module):
    """HardSigmoid (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.nn.functional.hardsigmoid(((x + r)))


batch_size = 192
dim = 96
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
