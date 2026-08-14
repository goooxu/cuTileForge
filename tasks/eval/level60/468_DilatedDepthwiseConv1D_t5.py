import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 5, conv)"""
    
    def __init__(self, in_channels=2, kernel_size=3, dilation=2):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Depthwise convolution: groups=in_channels, out_channels=in_channels
        # Using 1D dilated convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # depthwise
            dilation=dilation,
            bias=False
        )
        
        # Initialize with fixed values for determinism
        with torch.no_grad():
            nn.init.ones_(self.conv.weight)
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 2
SEQ_LENGTH = 8

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, seq_length)
    return [torch.ones(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    # Return configuration parameters for __init__
    return [IN_CHANNELS, 3, 2]  # in_channels, kernel_size, dilation