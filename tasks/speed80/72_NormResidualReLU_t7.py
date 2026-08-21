import torch
import torch.nn as nn


class Model(nn.Module):
    """NormResidualReLU (tier 7, norm)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        h = (x - mean) / torch.sqrt(var + self.eps)
        return torch.relu(h + r)


batch_size = 8192
dim = 20480

def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
