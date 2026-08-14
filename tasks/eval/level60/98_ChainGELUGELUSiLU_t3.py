import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUGELUSiLU (tier 3, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu((torch.nn.functional.gelu((torch.nn.functional.gelu((x.mean(dim=1, keepdim=True))))) * 1.7))


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
