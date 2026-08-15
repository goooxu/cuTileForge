import torch
import torch.nn as nn


class Model(nn.Module):
    """HardSigmoid (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardsigmoid(x)


batch_size = 9216
dim = 131072
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
