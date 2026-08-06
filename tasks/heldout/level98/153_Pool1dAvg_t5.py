import torch
import torch.nn as nn

"""Pool1dAvg (tier 5, pool)"""

# Shape constants
BATCH_SIZE = 8
IN_CHANNELS = 32
INPUT_LENGTH = 128
POOL_KERNEL_SIZE = 4
POOL_STRIDE = 4

class Model(nn.Module):
    """Pool1dAvg (tier 5, pool)"""

    def __init__(self, in_channels, input_length, kernel_size, stride):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        # Input shape: (batch_size, in_channels, input_length)
        # Apply average pooling along the sequence dimension
        output = self.pool(x)
        return output

def get_inputs():
    # Return a list of input tensors with the correct shape
    # Shape: (BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    # Return a list of arguments for __init__
    return [IN_CHANNELS, INPUT_LENGTH, POOL_KERNEL_SIZE, POOL_STRIDE]