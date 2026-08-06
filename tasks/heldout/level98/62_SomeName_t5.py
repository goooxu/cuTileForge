import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""
    def __init__(self):
        super(Model, self).__init__()
        self.eval()

    def forward(self, x):
        # Chain of 5 elementwise operations for throughput testing
        x = x + 1.0
        x = torch.sin(x)
        x = x * x
        x = torch.cos(x)
        x = x * 2.0
        return x

# Tensor shape constants for large tensor throughput testing
N = 8192
M = 8192

def get_inputs():
    # Return a list with one large tensor
    return [torch.ones(N, M, dtype=torch.float32)]

def get_init_inputs():
    # No configuration needed for this simple model
    return []