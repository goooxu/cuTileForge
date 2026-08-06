import torch
import torch.nn as nn

"""SimpleElementwiseChain (tier 5, elementwise)"""


class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        # Chain of elementwise operations: no learnable parameters needed
        pass

    def forward(self, x):
        # Chain of 5 elementwise operations on tensor x
        # No in-place modification of input, no randomness
        x1 = torch.tanh(x)
        x2 = torch.sin(x1)
        x3 = torch.cos(x2)
        x4 = torch.sqrt(torch.abs(x3) + 1e-5)
        x5 = torch.sigmoid(x4)
        return x5


# Module-level constants for shapes
INPUT_SIZE = 100
HIDDEN_SIZE = 200
OUTPUT_SIZE = 150
BATCH_SIZE = 32

# Additional size parameters
SHAPE_M = 256
SHAPE_N = 128


def get_inputs():
    """Returns a list containing the input tensor for forward pass"""
    # Create a tensor of appropriate size (use BATCH_SIZE for dimension matching)
    return [torch.randn(BATCH_SIZE, HIDDEN_SIZE)]


def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    # Model takes no configuration parameters
    return []