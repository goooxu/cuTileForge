import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.nn.functional.silu(torch.sigmoid(torch.nn.functional.hardswish(torch.nn.functional.gelu(torch.nn.functional.leaky_relu(torch.sigmoid(x), negative_slope=0.01))) + 1.5)))


batch_size = 6111
dim = 28000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
