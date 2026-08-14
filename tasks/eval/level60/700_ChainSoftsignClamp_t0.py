import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftsignClamp (tier 0, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.softsign(x), min=-1.0, max=1.0)


batch_size = 384
dim = 768
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
