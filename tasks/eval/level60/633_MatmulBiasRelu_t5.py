import torch
import torch.nn as nn

class Model(nn.Module):
    """MatmulBiasRelu (tier 5, matmul)"""

    def __init__(self, m, n, k):
        super(Model, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        
        # Initialize weight and bias with deterministic values
        self.weight = nn.Parameter(torch.randn(n, k, dtype=torch.float32) / torch.sqrt(torch.tensor(k, dtype=torch.float32)))
        self.bias = nn.Parameter(torch.zeros(n, dtype=torch.float32))

    def forward(self, x):
        # Matrix multiplication: (m x k) @ (k x n) = (m x n)
        out = torch.matmul(x, self.weight.t())
        
        # Add bias
        out = out + self.bias
        
        # Apply ReLU activation
        out = torch.relu(out)
        
        return out


# Module-level constants for shapes
M = 4096
N = 4096
K = 4096

def get_inputs():
    # Return input tensor for forward pass
    return [torch.randn(M, K, dtype=torch.float32)]

def get_init_inputs():
    # Return arguments for model initialization
    return [M, N, K]
_EVAL_MARK = 1
