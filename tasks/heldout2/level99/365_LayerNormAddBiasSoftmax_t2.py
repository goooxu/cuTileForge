import torch
import torch.nn as nn


class Model(nn.Module):
    """LayerNormAddBiasSoftmax (tier 2, norm)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.softmax(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)) + 1.5, dim=-1)


batch_size = 128
dim = 512

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
