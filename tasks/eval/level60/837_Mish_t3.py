import torch
import torch.nn as nn


class Model(nn.Module):
    """Mish (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(x)


batch_size = 20479
dim = 49153
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
