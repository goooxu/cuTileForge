import torch
import torch.nn as nn


class Model(nn.Module):
    """Mish (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(torch.nn.functional.softmin(x, dim=-1))


batch_size = 25
channels = 2
height = 9
width = 9
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
