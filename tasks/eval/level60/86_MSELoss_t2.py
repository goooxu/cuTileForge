import torch
import torch.nn as nn


class Model(nn.Module):
    """MSELoss (tier 2, loss)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        return torch.mean((predictions - targets) ** 2)


batch_size = 96
dim = 192
def get_inputs():
    return [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
