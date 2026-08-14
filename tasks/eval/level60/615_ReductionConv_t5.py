import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionConv (tier 5, reduction)"""
    
    def __init__(self, input_channels, output_channels):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Convolutional layer after reduction
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        
        # Reduction along the sequence length axis (dimension 2)
        # Using mean reduction for determinism
        x_reduced = x.mean(dim=2, keepdim=True)
        
        # Squeeze to remove the singleton dimension
        x_reduced = x_reduced.squeeze(2)
        
        # Reshape for 1D convolution (batch_size, input_channels, 1)
        x_reshaped = x_reduced.unsqueeze(2)
        
        # Apply convolution
        output = self.conv(x_reshaped)
        
        # Squeeze to remove the singleton dimension
        output = output.squeeze(2)
        
        return output


# Module-level constants for shapes
BATCH_SIZE = 48
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
SEQUENCE_LENGTH = 256

def get_inputs():
    """Generate input tensor for the model"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH)]

def get_init_inputs():
    """Generate initialization arguments for the model"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]