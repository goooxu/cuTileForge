import torch
import torch.nn as nn


class Model(nn.Module):
    """Softmin (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softmin(x, dim=-1)


batch_size = 2237
dim = 8939
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
