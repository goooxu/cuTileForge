import torch
import torch.nn as nn

"""
MaxPool1D (tier 5, pool)
"""

# Module-level constants for shape configuration
INPUT_CHANNELS = 2
INPUT_LENGTH = 8
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_CHANNELS = 2
OUTPUT_LENGTH = 4

class Model(nn.Module):
    """MaxPool1D (tier 5, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Create a MaxPool1d layer with the specified parameters
        self.max_pool = nn.MaxPool1d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING
        )
    
    def forward(self, x):
        # Apply max pooling to the input tensor
        return self.max_pool(x)

def get_inputs():
    # Create a sample input tensor with shape (batch_size, channels, length)
    # Using a small tensor for kernel porting exercise
    batch_size = 1
    return [torch.randn(batch_size, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    # No additional inputs needed for initialization
    return []
_EVAL_MARK = 1
