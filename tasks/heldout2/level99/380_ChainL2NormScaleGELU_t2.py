import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormScaleGELU (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu((x / torch.norm(x, p=2, dim=1, keepdim=True) * 1.7))


batch_size = 32
dim = 256

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
