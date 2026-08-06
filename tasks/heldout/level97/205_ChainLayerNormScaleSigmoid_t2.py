import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormScaleSigmoid (tier 2, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid((self.ln(x) * 1.7))


batch_size = 64
dim = 128

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
