import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(torch.nn.functional.gelu(torch.nn.functional.leaky_relu(x * 2.0, negative_slope=0.01)))


batch_size = 8192
dim = 24576

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
