import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainELUSoftplus (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(torch.nn.functional.elu(x, alpha=1.25))


batch_size = 96
dim = 192
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
