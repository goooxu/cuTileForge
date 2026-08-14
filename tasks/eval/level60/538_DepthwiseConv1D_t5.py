import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseConv1D (tier 5, conv)"""
    
    def __init__(self, in_channels, kernel_size, dilation, padding):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        
        # Create depthwise convolution layer
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=padding,
            bias=False
        )
        
        # Initialize weights with a deterministic pattern
        with torch.no_grad():
            # Use a simple deterministic initialization
            nn.init.constant_(self.depthwise_conv.weight, 0.1)
    
    def forward(self, x):
        return self.depthwise_conv(x)

# Module-level constants for shapes
INPUT_BATCH_SIZE = 2
INPUT_CHANNELS = 4
INPUT_LENGTH = 16
KERNEL_SIZE = 3
DILATION = 2
PADDING = 2

def get_inputs():
    """Create input tensor for the depthwise convolution."""
    # Create input with shape (batch_size, channels, length) for 1D convolution
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    """Create initialization parameters for the model."""
    return [INPUT_CHANNELS, KERNEL_SIZE, DILATION, PADDING]
_EVAL_MARK = 1
