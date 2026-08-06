import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

# Module-level constants for tensor shapes
N, C, H, W = 8, 16, 32, 32
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Average pooling layer
        self.pool = nn.AvgPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        # learnable parameters for elementwise operations
        self.alpha = nn.Parameter(torch.ones(1, C, H // POOL_KERNEL_SIZE, W // POOL_KERNEL_SIZE))
        self.beta = nn.Parameter(torch.zeros(1, C, H // POOL_KERNEL_SIZE, W // POOL_KERNEL_SIZE))
        
    def forward(self, x):
        # Apply pooling
        x = self.pool(x)
        # Apply elementwise operations (deterministic)
        x = self.alpha * x + self.beta
        return x


def get_inputs():
    """Return input tensors for the model."""
    return [torch.randn(N, C, H, W)]


def get_init_inputs():
    """Return arguments for __init__."""
    return []