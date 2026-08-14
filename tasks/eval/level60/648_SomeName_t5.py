import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Chain of elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = x * x  # square
        x = torch.sqrt(x)
        x = x - 0.5
        return x

# Module-level constants for shape
INPUT_SIZE = (64, 128, 64, 64)

def get_inputs():
    return [torch.randn(*INPUT_SIZE)]

def get_init_inputs():
    return []
_EVAL_MARK = 1
