import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampGELUTanhSigmoid (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.nn.functional.gelu((torch.clamp(x, min=-1.0, max=1.0)), approximate='tanh'))


batch_size = 12288
dim = 81920
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
