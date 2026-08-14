import torch
import torch.nn as nn


class Model(nn.Module):
    """SoftShrink (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softshrink(x, lambd=0.5)


batch_size = 1024
dim = 1048576

def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
