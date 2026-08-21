import torch
import torch.nn as nn


class Model(nn.Module):
    """SoftmaxChainSigmoid (tier 7, norm)"""

    def __init__(self, scale: float):
        super(Model, self).__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.softmax(x * self.scale, dim=1))


batch_size = 6144
dim = 24576
scale = 0.125

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [scale]
