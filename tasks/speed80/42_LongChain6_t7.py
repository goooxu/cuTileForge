import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.sigmoid(torch.nn.functional.softplus(torch.nn.functional.leaky_relu(torch.nn.functional.elu(x) * 2.0, negative_slope=0.01))))


batch_size = 2048
dim = 65536

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
