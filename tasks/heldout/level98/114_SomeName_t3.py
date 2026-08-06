import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

# Module-level constants for tensor shapes
INPUT_BATCH = 32
INPUT_CHANNELS = 64
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
POOL_KERNEL = 2
POOL_STRIDE = 2
SCALE_FACTOR = 1.5


class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=POOL_KERNEL,
            stride=POOL_STRIDE
        )
        
        # For batch norm, but we won't use it in forward to avoid randomness
        
        # Store parameters for elementwise operations
        self.scale = SCALE_FACTOR
        
    def forward(self, x):
        # Apply pooling layer
        x = self.pool(x)
        
        # Elementwise operations: scaling and bias addition
        x = x * self.scale
        
        # Add a small constant bias (not changing original input)
        x = x + 0.01
        
        # Elementwise ReLU activation
        x = torch.relu(x)
        
        return x


def get_inputs():
    """Return input tensors for the model"""
    return [
        torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    ]


def get_init_inputs():
    """Return arguments for __init__"""
    return []