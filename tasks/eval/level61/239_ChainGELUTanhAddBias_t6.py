import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUTanhAddBias (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.gelu(x, approximate='tanh')) + 1.5


batch_size = 2304
dim = 524288
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
