import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Chain of elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = torch.relu(x)
        x = x / (x.abs() + 1.0)
        return x

# Module-level constants for shape configuration
INPUT_SIZE = 1024
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 1024
BATCH_SIZE = 64

def get_inputs():
    # Return a list of tensors to pass to forward
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    return [x]

def get_init_inputs():
    # Return a list of arguments to pass to __init__
    return []