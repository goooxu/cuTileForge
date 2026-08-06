import torch
import torch.nn as nn

"""GroupedConv1D (tier 5, conv)"""

class Model(nn.Module):
    """GroupedConv1D (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, groups, input_length):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.input_length = input_length
        
        # Create grouped 1D transposed convolution
        # Each group handles in_channels // groups input channels and produces out_channels // groups output channels
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=False
        )
        
        # Ensure deterministic behavior
        self.transposed_conv.eval()
        
    def forward(self, x):
        return self.transposed_conv(x)

# Module-level constants for shapes
INPUT_BATCH_SIZE = 4
INPUT_CHANNELS = 8
INPUT_LENGTH = 32
KERNEL_SIZE = 5
OUTPUT_CHANNELS = 8
GROUPS = 2

def get_inputs():
    # Generate input tensor of shape (batch_size, in_channels, input_length)
    batch_size = INPUT_BATCH_SIZE
    in_channels = INPUT_CHANNELS
    input_length = INPUT_LENGTH
    
    # Create input tensor with fixed values for reproducibility
    x = torch.ones(batch_size, in_channels, input_length)
    
    return [x]

def get_init_inputs():
    return [
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
        KERNEL_SIZE,
        GROUPS,
        INPUT_LENGTH
    ]