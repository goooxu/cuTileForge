import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainScaleSoftsignReLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.nn.functional.softsign(((x * 2.0))))


batch_size = 20479
dim = 49153
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
