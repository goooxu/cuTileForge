import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.in_channels = in_channels
        
        # Define normalization layer
        self.norm = nn.BatchNorm2d(in_channels)
        
        # Define residual convolution for matching dimensions if needed
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Define activation
        self.activation = nn.ReLU(inplace=False)
        
        # Set BatchNorm to eval mode for deterministic behavior
        self.norm.eval()

    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply convolution to residual if dimensions match
        residual = self.residual_conv(residual)
        
        # Add residual
        x = x + residual
        
        # Apply activation
        x = self.activation(x)
        
        return x

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 32
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Return input tensors for the model forward pass."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization."""
    return [IN_CHANNELS]