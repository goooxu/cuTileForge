import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainGELUGELUTanhShrinkLeakyReLU (tier 2, matmul)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.tanhshrink((torch.nn.functional.gelu((torch.nn.functional.gelu((torch.matmul(A, B))))))), negative_slope=0.02)


M = 64
K = 128
N = 64

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]


def get_init_inputs():
    return []
