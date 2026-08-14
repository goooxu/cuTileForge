import torch
import torch.nn as nn


class Model(nn.Module):
    """ResidualClampGELUReLU (tier 3, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        return torch.relu(torch.nn.functional.gelu(torch.clamp((x + r), -2.0, 2.0)))


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim), torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
