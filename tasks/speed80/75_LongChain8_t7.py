import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.sigmoid(torch.nn.functional.gelu(torch.nn.functional.mish(torch.relu(torch.nn.functional.mish(torch.nn.functional.hardswish(x))) + 1.5))))


batch_size = 16384
dim = 12288

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
