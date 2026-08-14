import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

# Module-level constants for shape configuration
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 5
KERNEL_SIZE = 3
BATCH_SIZE = 2
HEIGHT = 8
WIDTH = 8

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE):
        super(Model, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=1)
        
        # Elementwise operations: ReLU and then batch normalization
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(output_channels)
        
        # Ensure batchnorm is in eval mode for deterministic behavior
        self.bn.eval()
    
    def forward(self, x):
        # Convolution followed by elementwise operations
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

def get_inputs():
    """Returns input tensor for the model."""
    # Create input tensor with shape (batch_size, channels, height, width)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns initialization arguments for the model."""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE]