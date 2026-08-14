import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxScaleGELUBias (tier 3, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.gelu((torch.softmax(x, dim=1) * 1.7)) + 0.3)


batch_size = 6144
dim = 3072
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
