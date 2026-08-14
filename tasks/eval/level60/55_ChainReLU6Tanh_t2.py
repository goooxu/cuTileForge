import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLU6Tanh (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.relu6(x))


batch_size = 96
dim = 192
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
