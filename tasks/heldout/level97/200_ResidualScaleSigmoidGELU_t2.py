import torch
import torch.nn as nn


class Model(nn.Module):
    """ResidualScaleSigmoidGELU (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.nn.functional.gelu(torch.sigmoid(((x + r) * 1.7)))


batch_size = 64
dim = 128

def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
