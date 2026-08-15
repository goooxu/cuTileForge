import torch
import torch.nn as nn


class Model(nn.Module):
    """ELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(x)


batch_size = 2560
dim = 524288
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
