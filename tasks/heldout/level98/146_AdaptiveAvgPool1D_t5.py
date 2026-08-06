import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool1D (tier 5, pool)"""

    def __init__(self, output_size, input_channels):
        super(Model, self).__init__()
        self.output_size = output_size
        self.input_channels = input_channels
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        return self.pool(x)


INPUT_CHANNELS = 256
OUTPUT_SIZE = 128
BATCH_SIZE = 32
SEQ_LENGTH = 4096

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    return [OUTPUT_SIZE, INPUT_CHANNELS]