import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for tensor shapes
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 512
BATCH_SIZE = 128
HEIGHT = 64
WIDTH = 64

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS):
        super(Model, self).__init__()
        
        # Create normalization layer
        self.norm = nn.BatchNorm2d(input_channels)
        
        # Create convolution layer for residual connection
        self.residual_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
        # Create main convolution layer
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        
        # Set normalization to evaluation mode for deterministic behavior
        self.norm.eval()
        
    def forward(self, x):
        # Store original input for residual connection
        residual = x
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply residual connection (project residual if needed)
        if residual.shape[1] != x.shape[1]:
            residual = self.residual_conv(residual)
        
        # Add residual
        x = x + residual
        
        # Apply activation (ReLU)
        x = torch.relu(x)
        
        return x

def get_inputs():
    """Create input tensors for the model."""
    # Create input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Create initialization arguments for the model."""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]