import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolLayer (tier 2, pool)"""

    def __init__(self, kernel_size, stride, padding, input_channels):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

# Module-level constants for tensor shapes
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
INPUT_CHANNELS = 16
BATCH_SIZE = 8
SEQUENCE_LENGTH = 64

def get_inputs():
    # Create input tensor with shape (batch_size, input_channels, sequence_length)
    # For MaxPool1d, input shape should be (batch, channels, sequence)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH)]

def get_init_inputs():
    return [KERNEL_SIZE, STRIDE, PADDING, INPUT_CHANNELS]