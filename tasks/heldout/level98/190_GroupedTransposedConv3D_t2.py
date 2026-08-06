import torch
import torch.nn as nn

"""GroupedTransposedConv3D (tier 2, conv)"""

# Module-level constants for shape configuration
BATCH_SIZE = 2
IN_CHANNELS = 16
OUT_CHANNELS = 32
GROUPS = 4
DILATION = 2
KERNEL_SIZE = 3
INPUT_DIM = 16

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size, dilation, input_dim):
        super().__init__()
        
        # Calculate output padding for transposed convolution to get desired output size
        output_padding = 0
        
        # Create transposed 3D convolution with grouping and dilation
        self.transposed_conv3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
            bias=False
        )
        
        # Initialize weights using kaiming_normal for better numerical stability
        nn.init.kaiming_normal_(self.transposed_conv3d.weight, mode='fan_out', nonlinearity='relu')
        
        # Set to eval mode for deterministic inference
        self.transposed_conv3d.eval()
    
    def forward(self, x):
        # Perform grouped transposed 3D convolution
        # This will produce an output with shape [BATCH_SIZE, OUT_CHANNELS, D, H, W]
        # where D, H, W are computed based on input size, kernel, stride, dilation, etc.
        return self.transposed_conv3d(x)

def get_inputs():
    """Create input tensors with the specified configuration."""
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_DIM, INPUT_DIM, INPUT_DIM)
    return [input_tensor]

def get_init_inputs():
    """Return the parameters needed to initialize the model."""
    return [IN_CHANNELS, OUT_CHANNELS, GROUPS, KERNEL_SIZE, DILATION, INPUT_DIM]