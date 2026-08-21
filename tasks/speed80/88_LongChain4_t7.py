import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.clamp(torch.sigmoid(torch.sigmoid(x)), min=-1.0, max=1.0))


batch_size = 3333
dim = 48000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
