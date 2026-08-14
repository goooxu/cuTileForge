import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 6
FEATURE_DIM = 8
REDUCE_DIM = 6

class Model(nn.Module):
    """SomeName (tier 3, reduction)"""

    def __init__(self):
        super().__init__()
        # No learnable parameters needed for this reduction example
        pass

    def forward(self, x):
        # Reduction along one axis (dim=2) followed by elementwise work
        # First reduce along the last dimension
        reduced = x.sum(dim=2)
        
        # Elementwise operations: multiply by a constant factor and add bias
        # This simulates some post-reduction processing
        result = reduced * 0.5 + 1.0
        
        return result

def get_inputs():
    """Create input tensor for the model"""
    # Create a tensor with shape (BATCH_SIZE, FEATURE_DIM, REDUCE_DIM)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM, REDUCE_DIM)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization"""
    return []