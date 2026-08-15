import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU (tier 3, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu((x * x))


batch_size = 28672
dim = 40960
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
