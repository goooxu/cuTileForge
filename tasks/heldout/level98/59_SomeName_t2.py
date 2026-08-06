import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

# Module-level constants for shapes
BATCH_SIZE = 64
INPUT_CHANNELS = 128
H = 256
W = 256

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        # This model performs a chain of elementwise operations
        # on a large tensor to measure throughput
        
    def forward(self, x):
        # Chain of elementwise operations on tensor x
        # Input shape: (BATCH_SIZE, INPUT_CHANNELS, H, W)
        
        # Operation 1: Add constant bias
        x = x + 0.1
        
        # Operation 2: Multiply by scale
        x = x * 1.5
        
        # Operation 3: Exponential
        x = torch.exp(x)
        
        # Operation 4: Natural log (undo exp for stability, but keep the chain)
        x = torch.log(x)
        
        # Operation 5: Power operation
        x = x ** 2.0
        
        # Operation 6: Square root
        x = torch.sqrt(x)
        
        # Operation 7: Absolute value
        x = torch.abs(x)
        
        # Operation 8: Clamp operation
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        return x


def get_inputs():
    # Generate input tensor with deterministic shape
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, H, W, dtype=torch.float32)
    return [x]


def get_init_inputs():
    # No initialization arguments needed
    return []