import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxScaleSquareGELUScale (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.gelu(((torch.softmax(x, dim=1) * 1.7) ** 2)) * 1.7)


batch_size = 192
dim = 96
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
