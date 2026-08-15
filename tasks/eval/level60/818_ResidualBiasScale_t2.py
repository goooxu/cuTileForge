import torch
import torch.nn as nn


class Model(nn.Module):
    """ResidualBiasScale (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return (((x + r) + 0.3) * 1.7)


batch_size = 16383
dim = 61441
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
