import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU6 (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu6(x)


batch_size = 3072
channels = 128
length = 64

def get_inputs():
    return [torch.randn(batch_size, channels, length)]


def get_init_inputs():
    return []
