import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolMaxElementwise (tier 3, pool)"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        # Max pooling layer
        pooled = torch.nn.functional.max_pool2d(x, 
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding)
        # Elementwise operations: add constant and apply relu
        result = pooled + 1.0
        result = torch.nn.functional.relu(result)
        return result

# Module-level constants for shape configuration
INPUT_BATCH_SIZE = 8
INPUT_CHANNELS = 64
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

def get_inputs():
    # Create input tensor for the model
    # Size: (batch_size, channels, height, width)
    x = torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    # Return arguments for model initialization
    return [KERNEL_SIZE, STRIDE, PADDING]