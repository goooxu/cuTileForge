import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.sigmoid(torch.nn.functional.elu(torch.nn.functional.gelu(torch.sigmoid(torch.nn.functional.gelu(x))) + 1.5))) * 2.0


batch_size = 5000
dim = 28000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
