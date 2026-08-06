import torch
import torch.nn as nn

"""SomeName (tier 5, elementwise)"""


class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Chain of 4 elementwise operations
        # Operation 1: Square the input
        out1 = x * x
        
        # Operation 2: Add a constant offset
        out2 = out1 + 1.0
        
        # Operation 3: Apply ReLU activation
        out3 = torch.relu(out2)
        
        # Operation 4: Compute square root
        out4 = torch.sqrt(out3)
        
        return out4


# Module-level constants for shapes
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
BATCH_SIZE = 8
CHANNELS = 3


def get_inputs():
    """Return a list of input tensors for forward pass."""
    # Create a tensor with medium size for testing
    # Shape: (BATCH_SIZE, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    x = torch.randn(BATCH_SIZE, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]


def get_init_inputs():
    """Return a list of arguments for __init__."""
    return []