import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormBiasSquareBiasBias (tier 2, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return ((((self.ln(x) + 0.3) ** 2) + 0.3) + 0.3)


batch_size = 32
dim = 256

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
