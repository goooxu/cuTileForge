import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 3
IN_CHANNELS = 4
HEIGHT = 12
WIDTH = 12
DEPTHWISE_GROUPS = 4
DEPTHWISE_KERNEL_SIZE = 3
POINTWISE_OUTPUT_CHANNELS = 6

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""
    
    def __init__(self):
        super().__init__()
        
        # Depthwise convolution: groups=IN_CHANNELS makes it depthwise
        self.depthwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS,
            kernel_size=DEPTHWISE_KERNEL_SIZE,
            padding=DEPTHWISE_KERNEL_SIZE // 2,
            groups=IN_CHANNELS
        )
        
        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=POINTWISE_OUTPUT_CHANNELS,
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Pointwise convolution
        x = self.pointwise_conv(x)
        
        return x

def get_inputs():
    """Return a list of input tensors for the model."""
    # Create input tensor with shape (batch_size, in_channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    # No arguments needed for initialization
    return []