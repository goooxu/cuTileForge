import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionMaxPool (tier 3, reduction)"""

    def __init__(self, input_channels, output_channels):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Linear layer to project after reduction
        self.projection = nn.Linear(input_channels, output_channels)
        
    def forward(self, x):
        # First reduce along dimension 1 (sequence length)
        # Using max reduction as it's deterministic
        reduced, _ = torch.max(x, dim=1)
        
        # Then apply elementwise linear transformation
        result = self.projection(reduced)
        
        return result

# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 32
BATCH_SIZE = 24
SEQ_LENGTH = 128

def get_inputs():
    # Create input tensor with shape [batch_size, seq_length, input_channels]
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_CHANNELS)
    return [x]

def get_init_inputs():
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]