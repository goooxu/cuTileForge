import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainAddBiasSoftplusELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu((torch.nn.functional.softplus((x + 1.5))), alpha=1.25)


batch_size = 15361
dim = 65535
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
