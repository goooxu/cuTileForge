import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.nn.functional.elu(torch.nn.functional.leaky_relu(x + 1.5, negative_slope=0.01) + 1.5) * 2.0)


batch_size = 7777
dim = 18000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
