import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMatmulReLUSquareClamp (tier 3, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.clamp((torch.relu(torch.matmul(A, B)) ** 2), -2.0, 2.0)


M = 4096
K = 2048
N = 512

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
