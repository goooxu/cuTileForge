import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(torch.nn.functional.hardswish(torch.nn.functional.silu(torch.nn.functional.elu(torch.nn.functional.softplus(torch.nn.functional.elu(torch.nn.functional.mish(x)) * 2.0)))))


batch_size = 6144
dim = 24576

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
