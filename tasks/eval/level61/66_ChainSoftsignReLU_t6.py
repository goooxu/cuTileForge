import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftsignReLU (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.relu(torch.nn.functional.softsign(x))


batch_size = 28672
dim = 40960
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
