import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU6 (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu6(x)


batch_size = 32
dim = 256

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
