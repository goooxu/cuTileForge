import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLUTanhHardSwish (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(torch.tanh(torch.relu(x)))


batch_size = 10240
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
