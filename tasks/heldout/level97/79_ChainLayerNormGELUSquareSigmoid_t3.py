import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormGELUSquareSigmoid (tier 3, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid((torch.nn.functional.gelu(self.ln(x)) ** 2))


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
