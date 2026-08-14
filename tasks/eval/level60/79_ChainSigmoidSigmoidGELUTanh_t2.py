import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSigmoidSigmoidGELUTanh (tier 2, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu((torch.sigmoid((torch.sigmoid(((x / torch.norm(x, p=2, dim=1, keepdim=True))))))), approximate='tanh')


batch_size = 96
dim = 192
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
