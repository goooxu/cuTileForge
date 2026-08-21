import torch
import torch.nn as nn


class Model(nn.Module):
    """LayerNormReLUGELUMaxLast (tier 7, norm)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.max(torch.nn.functional.gelu(torch.relu(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)))), dim=-1)[0]


batch_size = 8888
dim = 20000

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
