import torch
import torch.nn as nn


class Model(nn.Module):
    """LayerNormClampGELUSoftmax (tier 7, norm)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.softmax(torch.nn.functional.gelu(torch.clamp(((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)), min=-1.0, max=1.0)), dim=-1)


batch_size = 4096
dim = 40960

def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
