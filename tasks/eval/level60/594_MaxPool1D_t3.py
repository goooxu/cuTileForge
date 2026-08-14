import torch
import torch.nn as nn

"""MaxPool1D (tier 3, pool)"""

# Module-level constants for shape configuration
BATCH_SIZE = 6
IN_CHANNELS = 8
SEQ_LENGTH = 32
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

class Model(nn.Module):
    """MaxPool1D (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool1d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING
        )
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    """Generate input tensor for MaxPool1D"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    """Return empty list as __init__ takes no arguments"""
    return []