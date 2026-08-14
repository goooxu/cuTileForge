import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUReLU6Scale (tier 5, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return (torch.nn.functional.relu6((torch.nn.functional.gelu(torch.matmul(A, B))))) * 2.0


M = 8192
K = 8192
N = 512

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
