import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # BatchNorm - set to eval mode for deterministic behavior
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
        
        # Activation
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Store original input for residual connection
        residual = x
        
        # Apply convolution
        out = self.conv(x)
        
        # Apply batch normalization
        out = self.bn(out)
        
        # Add residual connection
        out = out + residual
        
        # Apply activation
        out = self.relu(out)
        
        return out


# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
OUT_CHANNELS = 4
KERNEL_SIZE = 3
HEIGHT = 8
WIDTH = 8

def get_inputs():
    """Return list of tensors to pass to forward method."""
    # Create input tensor with shape (batch, channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return list of arguments to pass to __init__ method."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]