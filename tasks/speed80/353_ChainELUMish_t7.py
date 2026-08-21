import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainELUMish (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(torch.nn.functional.elu(x))


batch_size = 8888
dim = 20000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
