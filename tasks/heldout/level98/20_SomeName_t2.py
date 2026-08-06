import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

# Configuration constants
NUM_CHANNELS = 1024
BATCH_SIZE = 256
INPUT_SIZE = 256

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, num_channels, input_size):
        super(Model, self).__init__()
        
        # Store dimensions for reference
        self.num_channels = num_channels
        self.input_size = input_size
        
        # LayerNorm configuration (applies normalization over channel dimension)
        self.norm = nn.LayerNorm([num_channels])
        
        # Convolution layer for residual connection (1x1 conv to match channel dimensions)
        self.residual_conv = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.residual_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.residual_conv.bias is not None:
            nn.init.zeros_(self.residual_conv.bias)
        
        # Put in eval mode to ensure deterministic behavior
        self.norm.eval()
        
    def forward(self, x):
        # x shape: [batch_size, num_channels, input_size]
        
        # Store original input for residual connection
        original = x
        
        # Apply normalization to input
        # Transpose for LayerNorm (expects [N, C, *] or [N, L] format)
        x_norm = x.transpose(1, 2)  # [batch_size, input_size, num_channels]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2)  # [batch_size, num_channels, input_size]
        
        # Apply convolution for residual connection
        residual = self.residual_conv(x)
        
        # Compute norm + residual
        combined = x_norm + residual
        
        # Apply activation
        output = self.activation(combined)
        
        return output

def get_inputs():
    """Return input tensors in the correct format for forward pass"""
    # Create a large tensor that will be used for computation
    # Shape: [batch_size, num_channels, input_size]
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, INPUT_SIZE)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [NUM_CHANNELS, INPUT_SIZE]