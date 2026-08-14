import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 6
CHANNELS = 16
HEIGHT = 48
WIDTH = 48
class Model(nn.Module):
    """RMSNorm (tier 5, norm)"""
    
    def __init__(self, normalized_shape=16):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-6
        
        # Create learnable parameters for normalization
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, x):
        # Compute RMS normalization
        # x: (batch_size, channels, height, width)
        # We normalize along the channel dimension (dim=1)
        
        # Calculate mean of squares along channel dimension
        mean_sq = (x ** 2).mean(dim=1, keepdim=True)
        
        # Normalize by RMS (root mean square)
        rms = torch.sqrt(mean_sq + self.eps)
        normalized = x / rms
        
        # Apply weight (broadcast across batch, height, width)
        # weight shape: (channels,) -> unsqueeze to (1, channels, 1, 1)
        weight_reshaped = self.weight.view(1, self.normalized_shape, 1, 1)
        return normalized * weight_reshaped

def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [CHANNELS]