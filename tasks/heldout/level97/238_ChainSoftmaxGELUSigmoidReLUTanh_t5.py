import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSoftmaxGELUSigmoidReLUTanh (tier 5, norm)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.tanh(torch.relu(torch.sigmoid(torch.nn.functional.gelu(torch.softmax(x, dim=1)))))


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
