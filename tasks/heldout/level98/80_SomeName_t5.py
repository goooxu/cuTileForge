import torch
import torch.nn as nn

"""
SomeName (tier 5, conv)
"""

# Module-level constants for tensor shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
BATCH_SIZE = 32
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3
PADDING = 1
STRIDE = 1

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=True):
        super(Model, self).__init__()
        
        # Create the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias
        )
        
        # Initialize weights with a deterministic pattern
        nn.init.constant_(self.conv.weight, 0.1)
        if bias:
            nn.init.constant_(self.conv.bias, 0.01)
        
        # Set to evaluation mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        # Apply convolution
        out = self.conv(x)
        
        # Apply ReLU activation
        out = torch.relu(out)
        
        return out

def get_inputs():
    """Return a list of tensors to pass to forward."""
    # Create input tensor with shape (batch_size, channels, height, width)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
        KERNEL_SIZE,
        PADDING,
        STRIDE,
        True  # bias
    ]