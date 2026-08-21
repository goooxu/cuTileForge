import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.nn.functional.softplus(torch.nn.functional.mish(torch.relu(torch.nn.functional.elu(x))) + 1.5))


batch_size = 3333
dim = 48000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
