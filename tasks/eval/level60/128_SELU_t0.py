import torch
import torch.nn as nn


class Model(nn.Module):
    """SELU (tier 0, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.selu(x)


batch_size = 769
dim = 384
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
