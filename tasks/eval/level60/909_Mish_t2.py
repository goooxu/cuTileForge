import torch
import torch.nn as nn


class Model(nn.Module):
    """Mish (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(x)


batch_size = 12288
dim = 81920
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
