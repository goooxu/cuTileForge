import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainELUReLU (tier 1, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.nn.functional.elu(x, alpha=1.25))


batch_size = 2304
dim = 524288
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
