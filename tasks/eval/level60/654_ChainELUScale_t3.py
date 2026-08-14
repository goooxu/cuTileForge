import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainELUScale (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.elu(x, alpha=1.25)) * 2.0


batch_size = 384
channels = 32
height = 16
width = 16
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
