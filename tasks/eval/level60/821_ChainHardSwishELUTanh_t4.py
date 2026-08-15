import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSwishELUTanh (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.elu((torch.nn.functional.hardswish(x)), alpha=1.25))


batch_size = 30719
dim = 32771
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
