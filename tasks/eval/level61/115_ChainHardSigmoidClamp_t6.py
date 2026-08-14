import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSigmoidClamp (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.hardsigmoid(x), min=-1.0, max=1.0)


batch_size = 6656
dim = 163840
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
