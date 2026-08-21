import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.clamp(torch.nn.functional.gelu(torch.relu(x) + 1.5) + 1.5, min=-1.0, max=1.0))


batch_size = 8192
dim = 20480

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
