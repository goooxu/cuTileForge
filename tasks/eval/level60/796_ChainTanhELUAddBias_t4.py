import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhELUAddBias (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.elu((torch.tanh(x)), alpha=1.25)) + 1.5


batch_size = 12288
dim = 81920
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
