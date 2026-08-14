import torch
import torch.nn as nn


class Model(nn.Module):
    """LogSigmoid (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.logsigmoid(x)


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
