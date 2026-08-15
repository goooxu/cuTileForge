import torch
import torch.nn as nn


class Model(nn.Module):
    """SiLU (tier 0, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(x)


batch_size = 24577
dim = 40961
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
