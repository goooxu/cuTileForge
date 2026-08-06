import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxBiasClamp (tier 3, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp((torch.softmax(x, dim=1) + 0.3), -2.0, 2.0)


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
