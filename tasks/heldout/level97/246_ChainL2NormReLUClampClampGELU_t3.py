import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormReLUClampClampGELU (tier 3, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.clamp(torch.clamp(torch.relu(x / torch.norm(x, p=2, dim=1, keepdim=True)), -2.0, 2.0), -2.0, 2.0))


batch_size = 2048
dim = 4096

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
