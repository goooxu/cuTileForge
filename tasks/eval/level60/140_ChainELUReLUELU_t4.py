import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainELUReLUELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu((torch.relu((torch.nn.functional.elu(x, alpha=1.25)))), alpha=1.25)


batch_size = 3073
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
