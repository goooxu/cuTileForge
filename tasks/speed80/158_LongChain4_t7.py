import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.elu(torch.nn.functional.elu(x)) * 2.0, min=-1.0, max=1.0)


batch_size = 16384
dim = 12288

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
