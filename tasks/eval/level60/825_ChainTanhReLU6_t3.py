import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhReLU6 (tier 3, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.nn.functional.relu6(((torch.tanh((x + r)) ** 2)))


batch_size = 12288
dim = 81920
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
