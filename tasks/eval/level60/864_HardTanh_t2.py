import torch
import torch.nn as nn


class Model(nn.Module):
    """HardTanh (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardtanh(x, min_val=-2.0, max_val=2.0)


batch_size = 5120
dim = 262144
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
