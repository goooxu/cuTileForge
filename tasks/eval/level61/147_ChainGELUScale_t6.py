import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUScale (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return (torch.nn.functional.gelu(x)) * 2.0


batch_size = 5120
dim = 262144
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
