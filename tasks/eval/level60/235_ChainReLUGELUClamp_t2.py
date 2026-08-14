import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLUGELUClamp (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.clamp(torch.nn.functional.gelu((torch.relu(((x + r))))), -2.0, 2.0)


batch_size = 96
dim = 192
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
