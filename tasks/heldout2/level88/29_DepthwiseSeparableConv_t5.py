import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=in_channels
        )
        
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


# Module-level constants for shape configuration
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
KERNEL_SIZE = 3
BATCH_SIZE = 8
HEIGHT = 224
WIDTH = 224

def get_inputs():
    """Generate input tensors for the model"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for Model initialization"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE]