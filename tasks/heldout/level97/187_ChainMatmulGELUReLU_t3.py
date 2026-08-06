import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainMatmulGELUReLU (tier 3, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.relu(torch.nn.functional.gelu(torch.matmul(A, B)))


M = 4096
K = 2048
N = 1024

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
