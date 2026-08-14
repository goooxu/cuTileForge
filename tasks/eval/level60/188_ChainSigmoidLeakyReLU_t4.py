import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSigmoidLeakyReLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.sigmoid(x), negative_slope=0.02)


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
