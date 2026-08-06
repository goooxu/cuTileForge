import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMatmulGELUClamp (tier 2, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.clamp(torch.nn.functional.gelu(torch.matmul(A, B)), -2.0, 2.0)


M = 128
K = 64
N = 128

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
