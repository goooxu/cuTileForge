import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, reduction)"""
    
    def __init__(self, input_channels, output_features):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.output_features = output_features
        
        # Linear layer for elementwise work after reduction
        self.fc = nn.Linear(input_channels, output_features)
        
        # Ensure deterministic behavior
        self.eval()
    
    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        # Reduction along seq_len dimension (axis 2) - elementwise sum
        reduced = x.sum(dim=2)  # shape: (batch_size, input_channels)
        
        # Elementwise work: linear transformation
        output = self.fc(reduced)  # shape: (batch_size, output_features)
        return output


# Module-level constants for shape configuration
INPUT_CHANNELS = 16
OUTPUT_FEATURES = 8
BATCH_SIZE = 4
SEQ_LEN = 6


def get_inputs():
    """Return a list of input tensors for forward pass."""
    # Create input tensor with shape (batch_size, input_channels, seq_len)
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LEN)
    return [x]


def get_init_inputs():
    """Return a list of arguments for model initialization."""
    return [INPUT_CHANNELS, OUTPUT_FEATURES]