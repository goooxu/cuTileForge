import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvolutionWithOperations (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, input_height, input_width):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()
        
        # Elementwise operations: batch norm and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolution
        out = self.conv(x)
        
        # Batch normalization
        out = self.bn(out)
        
        # ReLU activation
        out = self.relu(out)
        
        return out


# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
INPUT_HEIGHT = 48
INPUT_WIDTH = 48
BATCH_SIZE = 6
def get_inputs():
    """Return input tensors for the model"""
    # Create random input tensor with shape (batch_size, in_channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_HEIGHT, INPUT_WIDTH]