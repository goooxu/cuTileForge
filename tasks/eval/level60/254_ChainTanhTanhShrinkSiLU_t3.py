import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhTanhShrinkSiLU (tier 3, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.nn.functional.tanhshrink((torch.tanh(x.mean(dim=1, keepdim=True)))))


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
