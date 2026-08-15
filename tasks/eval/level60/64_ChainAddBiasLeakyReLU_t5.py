import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainAddBiasLeakyReLU (tier 5, norm)"""

    def __init__(self, dim: int, eps: float = 1e-4):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.logsumexp((((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt((x.var(dim=-1, keepdim=True, unbiased=False) + self.eps))) + 1.5), dim=-1), negative_slope=0.02)


batch_size = 193
dim = 513
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
