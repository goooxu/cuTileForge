import torch
import torch.nn as nn

"""SomeName (tier 5, pool)"""

# Module-level constants for shape configuration
INPUT_CHANNELS = 256
INPUT_HEIGHT = 1536
INPUT_WIDTH = 1536
BATCH_SIZE = 2
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

class Model(nn.Module):
    def __init__(self, input_channels, input_height, input_width, batch_size, pool_kernel_size, pool_stride):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        
        # Define pooling layer
        self.pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        
        # Define elementwise operation parameters
        self.scale = nn.Parameter(torch.ones(1, input_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
    
    def forward(self, x):
        # Apply pooling layer
        x = self.pool(x)
        
        # Apply elementwise operations (scale and bias)
        x = x * self.scale + self.bias
        
        # Additional elementwise operations for computational intensity
        x = torch.relu(x)
        x = x * x  # elementwise square
        x = torch.sqrt(torch.abs(x) + 1e-6)  # elementwise sqrt with numerical stability
        
        return x

def get_inputs():
    """Returns a list of input tensors for the model."""
    # Create input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(
        BATCH_SIZE, 
        INPUT_CHANNELS, 
        INPUT_HEIGHT, 
        INPUT_WIDTH
    )
    return [input_tensor]

def get_init_inputs():
    """Returns arguments for model initialization."""
    return [
        INPUT_CHANNELS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        BATCH_SIZE,
        POOL_KERNEL_SIZE,
        POOL_STRIDE
    ]