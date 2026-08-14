import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolMax1D (tier 5, pool)"""
    
    def __init__(self, kernel_size, stride, padding, input_channels, input_length):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        # Max pooling over time dimension (1D pooling)
        return torch.nn.functional.max_pool1d(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )

# Module-level constants for tensor shapes
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
INPUT_CHANNELS = 256
INPUT_LENGTH = 1024
BATCH_SIZE = 24
def get_inputs():
    # Create a large tensor suitable for measuring throughput
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    return [KERNEL_SIZE, STRIDE, PADDING, INPUT_CHANNELS, INPUT_LENGTH]