import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUAddBiasScale (tier 4, norm)"""

    def __init__(self, dim: int, eps: float = 1e-4):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return (torch.logsumexp((torch.nn.functional.silu(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt((x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)))) + 1.5), dim=-1)) * 2.0


batch_size = 192
dim = 512

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
