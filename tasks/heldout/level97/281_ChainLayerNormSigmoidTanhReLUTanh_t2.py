import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLayerNormSigmoidTanhReLUTanh (tier 2, norm)"""

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.relu(torch.tanh(torch.sigmoid(self.ln(x)))))


batch_size = 128
dim = 64

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
