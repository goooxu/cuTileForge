import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(torch.clamp(torch.nn.functional.softplus(torch.relu(torch.nn.functional.softplus(torch.nn.functional.softplus(torch.nn.functional.gelu(torch.clamp(x, min=-1.0, max=1.0)))))), min=-1.0, max=1.0))


batch_size = 6111
dim = 28000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
