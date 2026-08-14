import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanTanhTanh (tier 3, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.tanh(x.mean(dim=1, keepdim=True)))


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
