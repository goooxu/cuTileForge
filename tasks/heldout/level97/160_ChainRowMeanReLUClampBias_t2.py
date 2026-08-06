import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanReLUClampBias (tier 2, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.clamp(torch.relu(x.mean(dim=1, keepdim=True)), -2.0, 2.0) + 0.3)


batch_size = 64
dim = 128

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
