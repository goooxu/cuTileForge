import torch
import torch.nn as nn

"""
Pool1DLayer (tier 5, pool)
"""

# Module-level constants for shapes
BATCH_SIZE = 3
INPUT_CHANNELS = 4
INPUT_LENGTH = 16
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Using MaxPool1d for the pooling operation
        self.pool = nn.MaxPool1d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    # Create input tensor with shape (batch_size, channels, length)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    # No additional arguments needed for __init__
    return []