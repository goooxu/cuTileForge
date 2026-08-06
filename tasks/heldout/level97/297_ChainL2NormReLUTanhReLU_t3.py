import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormReLUTanhReLU (tier 3, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.tanh(torch.relu(x / torch.norm(x, p=2, dim=1, keepdim=True))))


batch_size = 4096
dim = 2048

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
