import torch
import torch.nn as nn

class Model(nn.Module):
    """LargeConvReLU (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        
        # Elementwise operations following convolution
        # Using ReLU and then Hardtanh to create two elementwise ops
        self.relu = nn.ReLU(inplace=False)  # Non-inplace to avoid modifying inputs
        self.hardtanh = nn.Hardtanh(min_val=0.0, max_val=20.0)
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Convolution
        out = self.conv(x)
        
        # First elementwise operation: ReLU
        out = self.relu(out)
        
        # Second elementwise operation: Hardtanh
        out = self.hardtanh(out)
        
        return out

# Module-level constants for tensor shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
BATCH_SIZE = 16
HEIGHT = 256
WIDTH = 256

def get_inputs():
    # Return a list with a single input tensor for the forward pass
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [input_tensor]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]