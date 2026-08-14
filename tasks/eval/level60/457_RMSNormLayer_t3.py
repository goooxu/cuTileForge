import torch
import torch.nn as nn

"""RMSNormLayer (tier 3, norm)"""

class Model(nn.Module):
    """RMSNormLayer (tier 3, conv)"""
    
    def __init__(self, num_features=64, eps=1e-4):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        # Compute RMS normalization
        # x shape: (batch_size, num_features, height, width)
        batch_size, num_features, height, width = x.shape
        
        # Compute sum of squares along feature dimension
        sq_sum = torch.sum(x * x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        
        # Compute mean of squares
        mean_sq = sq_sum / num_features  # (batch_size, 1, height, width)
        
        # Compute RMS: sqrt(mean_sq + eps)
        rms = torch.sqrt(mean_sq + self.eps)  # (batch_size, 1, height, width)
        
        # Normalize and scale
        normalized = x / rms  # (batch_size, num_features, height, width)
        
        # Reshape weight for broadcasting
        weight = self.weight.view(1, num_features, 1, 1)
        
        # Apply weight
        result = normalized * weight  # (batch_size, num_features, height, width)
        
        return result

# Module-level constants
BATCH_SIZE = 6
NUM_FEATURES = 64
HEIGHT = 12
WIDTH = 12
def get_inputs():
    """Returns a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [NUM_FEATURES, 1e-5]