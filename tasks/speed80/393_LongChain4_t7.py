import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.tanh(torch.nn.functional.hardswish(torch.nn.functional.softplus(x))))


batch_size = 4096
dim = 40960

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
