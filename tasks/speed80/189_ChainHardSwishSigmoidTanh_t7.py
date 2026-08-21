import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSwishSigmoidTanh (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.sigmoid(torch.nn.functional.hardswish(x)))


batch_size = 16384
dim = 12288

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
