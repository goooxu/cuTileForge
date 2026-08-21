import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.sigmoid(torch.nn.functional.elu(torch.sigmoid(torch.sigmoid(torch.nn.functional.gelu(x))))), negative_slope=0.01)


batch_size = 12288
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
