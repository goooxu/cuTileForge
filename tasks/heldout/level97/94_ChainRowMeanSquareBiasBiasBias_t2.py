import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanSquareBiasBiasBias (tier 2, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return ((((x.mean(dim=1, keepdim=True) ** 2) + 0.3) + 0.3) + 0.3)


batch_size = 32
dim = 256

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
