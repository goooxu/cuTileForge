import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain6 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.elu(torch.nn.functional.mish(torch.relu(torch.clamp(x, min=-1.0, max=1.0) + 1.5)) * 2.0)


batch_size = 3333
dim = 48000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
