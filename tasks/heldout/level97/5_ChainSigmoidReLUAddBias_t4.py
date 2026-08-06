import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSigmoidReLUAddBias (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.sigmoid(x)) + 1.5


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
