import torch
import torch.nn as nn

class Model(nn.Module):
    """Scale (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, dilation=1, padding=0):
        super(Model, self).__init__()
        
        # Validate that out_channels is divisible by groups
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()
    
    def forward(self, x):
        return (self.conv(x)) * 2.0


# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 256
KERNEL_SIZE = 3
GROUPS = 32
STRIDE = 1
DILATION = 2
PADDING = 1
BATCH_SIZE = 6
DEPTH = 24
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Returns a list of input tensors for the model."""
    # Create input tensor with shape (batch_size, in_channels, depth, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Returns a list of arguments for model initialization."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, GROUPS, STRIDE, DILATION, PADDING]