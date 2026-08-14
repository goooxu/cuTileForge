import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampClampTanhShrinkAddBias (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return (torch.nn.functional.tanhshrink((torch.clamp(torch.clamp((x + r), -2.0, 2.0), -2.0, 2.0)))) + 1.5


batch_size = 96
dim = 192
def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
