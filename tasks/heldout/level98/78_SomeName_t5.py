import torch
import torch.nn as nn

"""SomeName (tier 5, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        # Chain of elementwise operations:
        # 1. Square operation
        # 2. Addition with constant
        # 3. Square root operation  
        # 4. Division by constant
        # 5. Tanh activation (final)
        x = x * x  # square
        x = x + 1.0  # add constant
        x = torch.sqrt(x)  # square root
        x = x / 2.0  # divide by constant
        x = torch.tanh(x)  # tanh activation
        return x

# Module-level constants for shapes
BATCH_SIZE = 16
SEQ_LEN = 32
FEATURES = 64

def get_inputs():
    """Returns a list of tensors to pass to forward."""
    # Create input tensor with specified shape
    x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return []