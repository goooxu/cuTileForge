import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.gelu(torch.clamp(torch.nn.functional.leaky_relu(torch.nn.functional.softplus(torch.tanh(x + 1.5)), negative_slope=0.01), min=-1.0, max=1.0) + 1.5), min=-1.0, max=1.0)


batch_size = 8192
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
