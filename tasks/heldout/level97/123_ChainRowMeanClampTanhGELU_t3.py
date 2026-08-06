import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanClampTanhGELU (tier 3, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.tanh(torch.clamp(x.mean(dim=1, keepdim=True), -2.0, 2.0)))


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
