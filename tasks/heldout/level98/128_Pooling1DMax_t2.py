import torch
import torch.nn as nn

"""Pooling1DMax (tier 2, pool)"""

# Shape constants
BATCH_SIZE = 4
IN_CHANNELS = 8
INPUT_LENGTH = 64
POOL_KERNEL_SIZE = 4
POOL_STRIDE = 2

class Model(nn.Module):
    def __init__(self):
        """SomeName (tier 2, conv)"""
        super(Model, self).__init__()
        # This is actually a pooling layer despite the docstring pattern
        # Using a functional approach in forward to avoid in-place ops
        
    def forward(self, x):
        return torch.nn.functional.max_pool1d(x, 
                                              kernel_size=POOL_KERNEL_SIZE,
                                              stride=POOL_STRIDE)

def get_inputs():
    # Create a deterministic input tensor with values in range [0, 100]
    # Using integer values that when processed will give deterministic results
    x = torch.arange(BATCH_SIZE * IN_CHANNELS * INPUT_LENGTH, dtype=torch.float32).reshape(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    # Scale to reasonable range
    x = (x % 100).float()
    return [x]

def get_init_inputs():
    # For this model, initialization doesn't require parameters
    return []