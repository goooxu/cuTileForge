import torch
import torch.nn as nn


class Model(nn.Module):
    """Tanh (tier 1, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(x)


batch_size = 768
dim = 1572864
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
