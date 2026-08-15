import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampAddBiasHardSwish (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(((torch.clamp(x, min=-1.0, max=1.0) + 1.5)))


batch_size = 2560
dim = 524288
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
