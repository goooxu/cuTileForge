import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionConvModel (tier 2, reduction)"""
    
    def __init__(self, in_channels, out_channels, reduce_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduce_dim = reduce_dim
        
        # Conv layer after reduction
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        # Reduce along dimension specified by reduce_dim
        reduced = torch.sum(x, dim=self.reduce_dim, keepdim=True)
        
        # Reshape for 1D conv: (batch_size, 1, in_channels)
        batch_size = reduced.shape[0]
        channel_dim = reduced.shape[1]
        other_dims = reduced.shape[2:]
        
        # Reshape to (batch_size, 1, in_channels) for Conv1d
        if self.reduce_dim == 1:
            # Sum over channel dimension
            reshaped = reduced.view(batch_size, 1, -1)
        else:
            # For other cases, flatten appropriately
            reshaped = reduced.view(batch_size, 1, -1)
        
        # Apply convolution
        conv_result = self.conv(reshaped)
        
        return conv_result

# Module-level constants for shapes
BATCH_SIZE = 4
IN_CHANNELS = 8
OUT_CHANNELS = 16
REDUCE_DIM = 1

def get_inputs():
    # Create input tensor of shape (batch_size, in_channels, height, width)
    height = 4
    width = 4
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, height, width)
    return [x]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, REDUCE_DIM]