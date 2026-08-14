import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUGELUClamp (tier 5, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.clamp((torch.nn.functional.gelu((torch.nn.functional.gelu((torch.matmul(A, B))))) * 1.7), min=-1.0, max=1.0)


M = 4096
K = 16384
N = 1024

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
