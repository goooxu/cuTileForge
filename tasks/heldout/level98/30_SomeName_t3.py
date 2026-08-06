import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

class Model(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Model, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # First convolution: depthwise convolution
        # This convolution operates on each channel separately
        self.depthwise_conv = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=channels,
            bias=False
        )
        
        # Second convolution: pointwise convolution (1x1)
        # This convolution combines information across channels
        self.pointwise_conv = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=1, 
            padding=0, 
            groups=1,
            bias=False
        )
        
        # Set model to evaluation mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        # Apply pointwise convolution
        x = self.pointwise_conv(x)
        return x


# Module-level constants for tensor dimensions
INPUT_BATCH_SIZE = 16
INPUT_CHANNELS = 64
INPUT_HEIGHT = 256
INPUT_WIDTH = 256

def get_inputs():
    # Generate input tensor for forward pass
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Arguments to initialize the model
    return [INPUT_CHANNELS, 3]  # channels and kernel_size