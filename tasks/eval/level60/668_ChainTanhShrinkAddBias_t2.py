import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhShrinkAddBias (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.tanhshrink(x)) + 1.5


batch_size = 13
channels = 4
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
