import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""

# Module-level constants for shapes
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4
KERNEL_SIZE = 2
INPUT_SIZE = 9
class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=KERNEL_SIZE)
        # Define elementwise operation parameters
        self.scale = nn.Parameter(torch.tensor(1.5))
        self.shift = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x):
        # Apply pooling layer
        x = self.pool(x)
        # Apply elementwise operations: scale and shift
        x = x * self.scale + self.shift
        return x

def get_inputs():
    # Create input tensor with shape (batch_size, channels, height, width)
    batch_size = 2
    input_tensor = torch.randn(batch_size, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    return [input_tensor]

def get_init_inputs():
    # No additional inputs needed for __init__
    return []