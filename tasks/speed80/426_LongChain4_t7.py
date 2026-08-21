import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.nn.functional.mish(torch.nn.functional.hardswish(torch.sigmoid(x))))


batch_size = 16384
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
