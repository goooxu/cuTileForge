import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSigmoidTanh (tier 6, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.nn.functional.hardsigmoid(x))


batch_size = 7168
dim = 163840
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.rand(batch_size, dim, device=device)]


def get_init_inputs():
    return []
