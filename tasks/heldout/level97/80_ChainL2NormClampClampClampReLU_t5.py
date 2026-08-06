import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormClampClampClampReLU (tier 5, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.clamp(torch.clamp(torch.clamp(x / torch.norm(x, p=2, dim=1, keepdim=True), -2.0, 2.0), -2.0, 2.0), -2.0, 2.0))


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
