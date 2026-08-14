import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolMaxGlobalAvgPool (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Using global average pooling which is deterministic and inductive
        # No BatchNorm needed for this pure pooling example
    
    def forward(self, x):
        # Global max pooling followed by global average pooling
        # Both operations are deterministic and inductive
        max_pooled = torch.nn.functional.adaptive_max_pool2d(x, (1, 1))
        avg_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        # Element-wise operations: subtraction and then addition with scaling
        result = max_pooled - avg_pooled
        result = result * 2.0 + 1.0
        
        return result

# Module-level constants for shape configuration
BATCH_SIZE = 48
IN_CHANNELS = 256
HEIGHT = 84
WIDTH = 84
def get_inputs():
    """Returns list of input tensors for forward pass"""
    # Create a large tensor suitable for throughput measurement
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns list of arguments for __init__"""
    # No arguments needed for this simple model
    return []