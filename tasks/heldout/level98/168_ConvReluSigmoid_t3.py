import torch
import torch.nn as nn

"""ConvReluSigmoid (tier 3, conv)"""

# Module-level constants for shape configuration
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4
KERNEL_SIZE = 3
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
BATCH_SIZE = 2

class Model(nn.Module):
    """ConvReluSigmoid (tier 3, conv)"""
    
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE):
        super(Model, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    """Generate input tensors for the model."""
    return [
        torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    ]


def get_init_inputs():
    """Return arguments for model initialization."""
    return []