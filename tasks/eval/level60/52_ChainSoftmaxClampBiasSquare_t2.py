import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxClampBiasSquare (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return ((torch.clamp(torch.softmax(x, dim=1), -2.0, 2.0) + 0.3) ** 2)


batch_size = 49
dim = 385
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
