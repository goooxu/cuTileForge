import torch
import torch.nn as nn


class Model(nn.Module):
    """ELU (tier 0, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(x, alpha=1.0)


batch_size = 512
dim = 256

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
