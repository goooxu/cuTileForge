import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.softplus(torch.nn.functional.softplus(torch.clamp(torch.nn.functional.leaky_relu(torch.nn.functional.hardswish(torch.nn.functional.mish(x)) + 1.5, negative_slope=0.01), min=-1.0, max=1.0))))


batch_size = 8192
dim = 20480

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
