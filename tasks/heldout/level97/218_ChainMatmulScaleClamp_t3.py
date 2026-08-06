import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMatmulScaleClamp (tier 3, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.clamp((torch.matmul(A, B) * 1.7), -2.0, 2.0)


M = 2048
K = 4096
N = 512

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
