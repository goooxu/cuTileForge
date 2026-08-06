import torch
import torch.nn as nn

"""DilatedConv3D (tier 2, conv)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 64
OUT_CHANNELS = 128
DEPTH = 32
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
DILATION = 2
PADDING = DILATION

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, depth, height, width, kernel_size, dilation):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation
        
        # Create a dilated 3D convolution layer
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation
        )
        
        # Make BatchNorm eval to ensure deterministic behavior
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.batch_norm.eval()
        
        # Use ReLU activation
        self.relu = nn.ReLU()
        
        # Initialize weights with a deterministic approach
        # We don't use Kaiming initialization as it involves randomness in some versions
        # Instead, we'll use a simple constant initialization for deterministic behavior
        with torch.no_grad():
            nn.init.constant_(self.conv3d.weight, 0.1)
            if self.conv3d.bias is not None:
                nn.init.constant_(self.conv3d.bias, 0.0)

    def forward(self, x):
        # First conv operation
        out = self.conv3d(x)
        
        # Apply batch normalization (which is in eval mode, so deterministic)
        out = self.batch_norm(out)
        
        # Apply ReLU
        out = self.relu(out)
        
        return out


def get_inputs():
    """Returns input tensors for the forward pass"""
    # Input shape: (batch, channels, depth, height, width)
    input_tensor = torch.randn(
        BATCH_SIZE, 
        IN_CHANNELS, 
        DEPTH, 
        HEIGHT, 
        WIDTH
    )
    return [input_tensor]


def get_init_inputs():
    """Returns initialization arguments for the model"""
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        KERNEL_SIZE,
        DILATION
    ]