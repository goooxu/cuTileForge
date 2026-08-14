import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanSquareBiasScaleTanh (tier 2, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh((((x.mean(dim=1, keepdim=True) ** 2) + 0.3) * 1.7))


batch_size = 128
dim = 64

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
