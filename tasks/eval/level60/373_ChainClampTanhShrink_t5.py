import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampTanhShrink (tier 5, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.tanhshrink((torch.clamp((x / torch.norm(x, p=2, dim=1, keepdim=True)), -2.0, 2.0)))


batch_size = 2236
dim = 8944
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
