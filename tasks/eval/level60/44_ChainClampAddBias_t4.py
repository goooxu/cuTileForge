import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampAddBias (tier 4, norm)"""

    def __init__(self, dim: int, eps: float = 1e-4):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return (torch.clamp(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt((x.var(dim=-1, keepdim=True, unbiased=False) + self.eps))), min=-1.0, max=1.0)) + 1.5


batch_size = 192
dim = 256

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
