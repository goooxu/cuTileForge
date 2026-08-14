import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMatmulBiasBias (tier 2, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return ((torch.matmul(A, B) + 0.3) + 0.3)


M = 32
K = 256
N = 64

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []

_EVAL_MARK = 1
