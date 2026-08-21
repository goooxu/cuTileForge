import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.nn.functional.elu(torch.nn.functional.leaky_relu(torch.tanh(torch.sigmoid(torch.nn.functional.leaky_relu(x * 2.0, negative_slope=0.01))), negative_slope=0.01)) + 1.5)


batch_size = 12288
dim = 12288

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
