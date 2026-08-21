import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.clamp(torch.sigmoid(torch.nn.functional.gelu(torch.clamp(torch.tanh(torch.relu(x + 1.5) + 1.5), min=-1.0, max=1.0))), min=-1.0, max=1.0)


batch_size = 3073
dim = 40960

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
