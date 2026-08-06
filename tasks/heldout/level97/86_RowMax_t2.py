import torch
import torch.nn as nn


class Model(nn.Module):
    """RowMax (tier 2, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.max(x, dim=1)[0]


batch_size = 32
dim = 256

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
