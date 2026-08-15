import torch
import torch.nn as nn


class Model(nn.Module):
    """Scale (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (x + (1.5 * 2.0))


batch_size = 16383
dim = 61441
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
