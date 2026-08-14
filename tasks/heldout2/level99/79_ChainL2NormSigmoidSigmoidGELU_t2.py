import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainL2NormSigmoidSigmoidGELU (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.sigmoid(torch.sigmoid(x / torch.norm(x, p=2, dim=1, keepdim=True))))


batch_size = 64
dim = 128

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
