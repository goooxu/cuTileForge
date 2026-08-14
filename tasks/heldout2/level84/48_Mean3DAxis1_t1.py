import torch
import torch.nn as nn


class Model(nn.Module):
    """Mean3DAxis1 (tier 1, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.mean(x, dim=1)


batch_size = 4
channels = 16
length = 256

def get_inputs():
    return [torch.rand(batch_size, channels, length)]


def get_init_inputs():
    return []
