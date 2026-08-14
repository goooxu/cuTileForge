import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Model, self).__init__()
        
        # Configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )
        
        # Normalization layer
        self.norm = nn.BatchNorm2d(out_channels)
        
        # Ensure deterministic behavior for BatchNorm
        self.norm.eval()
        
    def forward(self, x):
        # Initial normalization
        x = self.norm(x)
        
        # Residual connection: add input to output
        residual = x
        
        # Convolution operation
        out = self.conv(x)
        
        # Activation function
        out = torch.relu(out)
        
        # Add residual connection
        out = out + residual
        
        return out


# Module-level constants for shapes
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 16
KERNEL_SIZE = 3
BATCH_SIZE = 1
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Return a list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE]