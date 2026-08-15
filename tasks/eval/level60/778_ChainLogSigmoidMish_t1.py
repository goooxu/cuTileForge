import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainLogSigmoidMish (tier 1, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mish(torch.nn.functional.logsigmoid(x))


batch_size = 35839
dim = 28673
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
