import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

NORM_CHANNELS = 8
RESIDUAL_CHANNELS = 8
TENSOR_SIZE = (1, 8, 4, 4)

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, norm_channels, residual_channels):
        super(Model, self).__init__()
        self.norm_channels = norm_channels
        self.residual_channels = residual_channels
        
        # LayerNorm for normalization (works on channel dimension)
        self.layer_norm = nn.LayerNorm([norm_channels, 4, 4])
        
        # Conv1x1 for residual connection if needed
        self.residual_conv = nn.Conv2d(residual_channels, norm_channels, 1) if residual_channels != norm_channels else None
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x, residual):
        # Normalize the input
        normalized = self.layer_norm(x)
        
        # Add residual
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # Ensure shapes match for addition
        if normalized.shape != residual.shape:
            # If residual needs to be resized, use interpolation
            residual = nn.functional.interpolate(residual, size=normalized.shape[2:], mode='nearest')
        
        out = normalized + residual
        
        # Apply activation
        out = self.activation(out)
        
        return out


def get_inputs():
    """Generate input tensors for the model."""
    x = torch.randn(TENSOR_SIZE)
    residual = torch.randn(TENSOR_SIZE)
    return [x, residual]


def get_init_inputs():
    """Generate initialization arguments for the model."""
    return [NORM_CHANNELS, RESIDUAL_CHANNELS]