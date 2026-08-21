import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSwishHardSwishTanh (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.hardswish(torch.nn.functional.hardswish(x)))


batch_size = 9999
dim = 16000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
