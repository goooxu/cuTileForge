import torch
import torch.nn as nn


class Model(nn.Module):
    """ResidualScaleGELUTanh (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.tanh(torch.nn.functional.gelu(((x + r) * 1.7)))


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
