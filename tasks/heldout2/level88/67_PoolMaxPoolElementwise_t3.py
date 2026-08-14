import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolMaxPoolElementwise (tier 3, pool)"""

    def __init__(self, input_channels, output_channels):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Elementwise operation: convolution to match channels
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
        # Elementwise addition
        self.add = torch.add

    def forward(self, x):
        # Apply pooling
        pooled = self.pool(x)
        
        # Apply convolution to match output channels
        conv_output = self.conv(pooled)
        
        # Elementwise addition with itself (as a simple elementwise operation)
        result = self.add(conv_output, conv_output)
        
        return result

# Module-level constants for shape configuration
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
BATCH_SIZE = 8
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Generate input tensor for the model"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]