import torch
import torch.nn as nn

class Model(nn.Module):
    """AvgPool1d (tier 2, pool)"""
    
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super(Model, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        # Ensure deterministic behavior by setting to eval mode
        self.pool.eval()
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants
BATCH_SIZE = 4
SEQ_LENGTH = 64
IN_CHANNELS = 32
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
CEIL_MODE = False

def get_inputs():
    """Generate deterministic input tensor"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    """Return arguments for __init__"""
    return [KERNEL_SIZE, STRIDE, PADDING, CEIL_MODE]