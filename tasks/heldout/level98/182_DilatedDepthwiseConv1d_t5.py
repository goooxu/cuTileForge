import torch
import torch.nn as nn

"""DilatedDepthwiseConv1d (tier 5, conv)"""

# Module-level constants for shapes
INPUT_CHANNELS = 3
INPUT_LENGTH = 16
KERNEL_SIZE = 3
DILATION = 2
OUTPUT_CHANNELS = 3  # Must equal INPUT_CHANNELS for depthwise

class Model(nn.Module):
    """DilatedDepthwiseConv1d (tier 5, conv)"""
    
    def __init__(self, in_channels, input_length, kernel_size, dilation, out_channels):
        super(Model, self).__init__()
        # Create padding to maintain input length when using dilation
        # For depthwise conv with dilation, padding = (kernel_size - 1) * dilation // 2
        padding = ((kernel_size - 1) * dilation) // 2
        
        # Use Conv1d with groups=in_channels for depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True
        )
        
        # Initialize weights with deterministic values
        torch.nn.init.constant_(self.conv.weight, 0.1)
        torch.nn.init.zeros_(self.conv.bias)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, input_length)
        # For depthwise conv, out_channels must equal in_channels
        return self.conv(x)

def get_inputs():
    """Generate input tensors for the model"""
    batch_size = 2
    x = torch.randn(batch_size, INPUT_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """Return arguments for __init__"""
    return [INPUT_CHANNELS, INPUT_LENGTH, KERNEL_SIZE, DILATION, OUTPUT_CHANNELS]