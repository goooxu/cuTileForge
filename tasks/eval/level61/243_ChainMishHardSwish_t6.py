import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMishHardSwish (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(torch.nn.functional.mish(x))


batch_size = 3072
dim = 327680
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
