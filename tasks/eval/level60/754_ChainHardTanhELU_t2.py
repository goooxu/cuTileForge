import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardTanhELU (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.nn.functional.hardtanh(x, min_val=-2.0, max_val=2.0))


batch_size = 12
channels = 4
height = 8
width = 8
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
