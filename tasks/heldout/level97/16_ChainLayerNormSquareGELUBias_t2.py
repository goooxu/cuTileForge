import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormSquareGELUBias (tier 2, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.gelu((self.ln(x) ** 2)) + 0.3)


batch_size = 128
dim = 64

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
