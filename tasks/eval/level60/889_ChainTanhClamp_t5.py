import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhClamp (tier 5, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.tanh(x), min=-1.0, max=1.0)


batch_size = 3584
dim = 294912
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
