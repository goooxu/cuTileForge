import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv2D (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
        
        # Create a dilated convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
        # Initialize weights for reproducibility
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        # Apply dilated convolution
        out = self.conv(x)
        return out

# Module-level constants for shapes
IN_CHANNELS = 128
OUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 2
DILATION = 2
BATCH_SIZE = 24
HEIGHT = 192
WIDTH = 192
def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]