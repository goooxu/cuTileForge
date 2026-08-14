import torch
import torch.nn as nn


class Model(nn.Module):
    """GELUTanh (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu((((x / torch.norm(x, p=2, dim=1, keepdim=True)) * 1.7)), approximate='tanh')


batch_size = 48
dim = 384
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
