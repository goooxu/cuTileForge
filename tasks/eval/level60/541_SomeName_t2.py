import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4
BATCH_SIZE = 3
TENSOR_HEIGHT = 3
TENSOR_WIDTH = 3

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS):
        super(Model, self).__init__()
        
        # Initialize convolution layer for residual connection
        self.residual_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
        # Initialize batch normalization
        self.bn = nn.BatchNorm2d(output_channels)
        
        # Set to eval mode for deterministic behavior
        self.bn.eval()
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Apply batch normalization
        x = self.bn(x)
        
        # Compute residual connection using 1x1 convolution
        residual = self.residual_conv(x)
        
        # Add residual and apply activation
        out = x + residual
        out = self.relu(out)
        
        return out

def get_inputs():
    # Generate input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TENSOR_HEIGHT, TENSOR_WIDTH)
    return [x]

def get_init_inputs():
    # Return configuration for model initialization
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]