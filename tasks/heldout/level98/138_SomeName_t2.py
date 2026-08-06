import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Chain of 4 elementwise operations
        y1 = torch.sin(x)
        y2 = torch.cos(y1)
        y3 = torch.tanh(y2)
        y4 = y3 * y3  # elementwise multiplication
        return y4


# Module-level constants for shape
INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64

def get_inputs():
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    return []