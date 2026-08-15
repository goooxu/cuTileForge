import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUSigmoidELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.sigmoid((torch.nn.functional.silu(x))))


batch_size = 9216
dim = 131072
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
