import torch
import torch.nn as nn


class Model(nn.Module):
    """ResidualClampSigmoidScale (tier 3, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return (torch.sigmoid(torch.clamp((x + r), -2.0, 2.0)) * 1.7)


batch_size = 24577
dim = 40961
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
