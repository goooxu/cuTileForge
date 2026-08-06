import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchedMatmul (tier 5, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.bmm(A, B)


batch_size = 64
M = 8192
K = 8192
N = 1024

def get_inputs():
    return [torch.rand(batch_size, M, K), torch.rand(batch_size, K, N)]


def get_init_inputs():
    return []
