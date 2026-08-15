import torch
import torch.nn as nn


class Model(nn.Module):
    """Softsign (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softsign(x)


batch_size = 1280
dim = 1048576
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
