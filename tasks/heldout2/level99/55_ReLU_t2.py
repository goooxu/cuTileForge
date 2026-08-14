import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(x)


batch_size = 64
dim = 128

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
