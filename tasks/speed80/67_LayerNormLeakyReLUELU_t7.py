import torch
import torch.nn as nn


class Model(nn.Module):
    """LayerNormLeakyReLUELU (tier 7, norm)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.nn.functional.leaky_relu(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)), negative_slope=0.01))


batch_size = 4096
dim = 40960

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
