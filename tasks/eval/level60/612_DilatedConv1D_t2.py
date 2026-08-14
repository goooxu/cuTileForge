import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv1D (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 2
OUT_CHANNELS = 4
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 2
SEQ_LENGTH = 10

def get_inputs():
    """Return list of input tensors"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]