import torch
import torch.nn as nn


class Model(nn.Module):
    """LongChain8 (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.tanh(torch.nn.functional.mish(torch.nn.functional.softplus(torch.clamp(torch.sigmoid(torch.nn.functional.hardswish(torch.nn.functional.gelu(x))), min=-1.0, max=1.0)))), negative_slope=0.01)


batch_size = 12288
dim = 16384

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
