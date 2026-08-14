import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSwishReLU6 (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu6((torch.nn.functional.hardswish(x)))


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
