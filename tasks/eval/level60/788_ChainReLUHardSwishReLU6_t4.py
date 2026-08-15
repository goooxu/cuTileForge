import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainReLUHardSwishReLU6 (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu6((torch.nn.functional.hardswish((torch.relu(x)))))


batch_size = 31745
dim = 40959
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
