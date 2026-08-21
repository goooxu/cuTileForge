import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain4 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(torch.nn.functional.mish(torch.relu(x))) + 1.5


batch_size = 3073
dim = 40960

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
