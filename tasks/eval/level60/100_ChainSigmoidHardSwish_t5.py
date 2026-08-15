import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSigmoidHardSwish (tier 5, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.nn.functional.hardswish((torch.sigmoid(torch.matmul(A, B)) * 1.7))


M = 4097
K = 16385
N = 1025
def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
