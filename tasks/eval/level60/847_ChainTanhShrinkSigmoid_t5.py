import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhShrinkSigmoid (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.sigmoid(torch.nn.functional.tanhshrink((((x + r) + 0.3))))


batch_size = 12288
dim = 81920
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
