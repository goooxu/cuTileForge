import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormTanhReLUBiasScale (tier 5, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return ((torch.relu(torch.tanh(x / torch.norm(x, p=2, dim=1, keepdim=True))) + 0.3) * 1.7)


batch_size = 4096
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
