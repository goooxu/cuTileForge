import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhHardSigmoidClamp (tier 2, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.nn.functional.hardsigmoid((torch.tanh(self.ln(x)))), min=-1.0, max=1.0)


batch_size = 192
dim = 64

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
