import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSigmoidReLU (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.sigmoid(x))


batch_size = 2500
dim = 80000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
