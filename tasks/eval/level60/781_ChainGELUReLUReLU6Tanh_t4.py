import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUReLUReLU6Tanh (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.relu6((torch.relu((torch.nn.functional.gelu(x))))))


batch_size = 2304
dim = 524288
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
