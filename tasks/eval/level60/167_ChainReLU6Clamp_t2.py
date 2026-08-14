import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLU6Clamp (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.relu6(x), min=-1.0, max=1.0)


batch_size = 24
channels = 2
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
