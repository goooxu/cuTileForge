import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(torch.nn.functional.softplus(torch.tanh(torch.nn.functional.softplus(torch.nn.functional.hardswish(torch.tanh(torch.tanh(x))) * 2.0))))


batch_size = 2048
dim = 65536

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
