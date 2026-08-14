import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, hidden_channels):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Convolutional layer for projection
        self.conv_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # Batch normalization - set to eval mode for deterministic behavior
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.bn.eval()
        
        # Activation function
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # Apply convolution projection
        x_proj = self.conv_proj(x)
        
        # Apply batch normalization
        x_norm = self.bn(x_proj)
        
        # Residual addition (using original input as residual after projection)
        # For this to work, we need to ensure x has the right dimensions
        # We'll use adaptive pooling if needed, but for simplicity, assume matching dimensions
        x_residual = x_proj  # In a real scenario, this would be a projection of x
        
        # Add residual
        x_out = x_norm + x_residual
        
        # Apply activation
        x_out = self.relu(x_out)
        
        return x_out

# Module-level constants for shapes
BATCH_SIZE = 12
IN_CHANNELS = 64
HIDDEN_CHANNELS = 128
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [IN_CHANNELS, HIDDEN_CHANNELS]