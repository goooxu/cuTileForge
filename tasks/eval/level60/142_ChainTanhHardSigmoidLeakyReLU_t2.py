import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhHardSigmoidLeakyReLU (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.hardsigmoid((torch.tanh((x + r)))), negative_slope=0.02)


batch_size = 48
dim = 384
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
