import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""


class Model(nn.Module):
    def __init__(self, channels, input_size):
        super(Model, self).__init__()
        self.channels = channels
        self.input_size = input_size
        
        # Layer normalization for the channel dimension
        self.norm = nn.LayerNorm(channels)
        
        # 1D convolution for feature processing
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        # Element-wise activation
        self.activation = nn.ReLU()
        
        # Set evaluation mode for deterministic behavior
        self.norm.eval()
        self.conv.eval()

    def forward(self, x):
        # x shape: (batch_size, channels, input_size)
        
        # Transpose to (batch_size, input_size, channels) for LayerNorm
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Transpose back to (batch_size, channels, input_size)
        x = x.transpose(1, 2)
        
        # Apply convolution
        conv_output = self.conv(x)
        
        # Add residual connection
        residual = x + conv_output
        
        # Apply activation function
        output = self.activation(residual)
        
        return output


# Module-level constants for tensor shapes
BATCH_SIZE = 12
CHANNELS = 64
INPUT_SIZE = 128

def get_inputs():
    """Generate input tensors for the model forward pass."""
    # Create a tensor with shape (batch_size, channels, input_size)
    x = torch.randn(BATCH_SIZE, CHANNELS, INPUT_SIZE)
    return [x]

def get_init_inputs():
    """Generate arguments for model initialization."""
    return [CHANNELS, INPUT_SIZE]