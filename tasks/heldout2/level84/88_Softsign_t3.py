import torch
import torch.nn as nn


class Model(nn.Module):
    """Softsign (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softsign(x)


batch_size = 1024
channels = 32
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
