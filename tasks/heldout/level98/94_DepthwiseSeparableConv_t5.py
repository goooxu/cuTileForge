import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels to process each channel separately
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # Initialize weights
        nn.init.kaiming_normal_(self.depthwise.weight)
        nn.init.kaiming_normal_(self.pointwise.weight)
        
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        # Pointwise convolution
        x = self.pointwise(x)
        return x

# Module-level constants for shapes
INPUT_BATCH = 1
INPUT_CHANNELS = 64
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
KERNEL_SIZE = 3

def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [
        torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    ]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [
        INPUT_CHANNELS,
        INPUT_CHANNELS * 2,  # Output channels (double the input)
        KERNEL_SIZE
    ]