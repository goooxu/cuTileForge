import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Chain of four elementwise operations: x → x² → sin(x²) → log(1 + |sin(x²)|) → exp(...)^2
        x1 = x * x  # x²
        x2 = torch.sin(x1)  # sin(x²)
        x3 = torch.log(1 + torch.abs(x2))  # log(1 + |sin(x²)|)
        x4 = torch.exp(x3)  # exp(log(1 + |sin(x²)|))
        result = x4 * x4  # [exp(log(1 + |sin(x²)|))]²
        return result

# Module-level constants for shape
BATCH_SIZE = 2
INPUT_CHANNELS = 4
HEIGHT = 8
WIDTH = 8

def get_inputs():
    """Return a list of tensors to pass to forward"""
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """Return arguments to pass to __init__"""
    return []