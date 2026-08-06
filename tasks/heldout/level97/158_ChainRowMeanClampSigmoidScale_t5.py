import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainRowMeanClampSigmoidScale (tier 5, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.sigmoid(torch.clamp(x.mean(dim=1, keepdim=True), -2.0, 2.0)) * 1.7)


batch_size = 4096
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
