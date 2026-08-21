import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.nn.functional.silu(torch.nn.functional.gelu(torch.nn.functional.silu(torch.tanh(x)) * 2.0)))


batch_size = 4096
dim = 32768

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
