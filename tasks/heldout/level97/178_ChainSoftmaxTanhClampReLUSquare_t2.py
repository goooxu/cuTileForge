import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxTanhClampReLUSquare (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.relu(torch.clamp(torch.tanh(torch.softmax(x, dim=1)), -2.0, 2.0)) ** 2)


batch_size = 64
dim = 128

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
