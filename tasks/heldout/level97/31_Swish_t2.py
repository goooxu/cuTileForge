import torch
import torch.nn as nn


class Model(nn.Module):
    """Swish (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


batch_size = 128
dim = 64

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
