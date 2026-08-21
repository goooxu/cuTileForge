import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.hardswish(torch.nn.functional.leaky_relu(torch.nn.functional.leaky_relu(torch.nn.functional.gelu(torch.nn.functional.mish(torch.sigmoid(x))), negative_slope=0.01), negative_slope=0.01))


batch_size = 9999
dim = 16000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
