import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormClampTanh (tier 5, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.clamp(self.ln(x), -2.0, 2.0))


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
