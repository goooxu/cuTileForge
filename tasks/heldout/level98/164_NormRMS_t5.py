import torch
import torch.nn as nn

"""
NormRMS (tier 5, norm)
"""

# Module-level constants for shape configuration
BATCH_SIZE = 16
SEQ_LEN = 4096
NUM_FEATURES = 4096

class Model(nn.Module):
    """NormRMS (tier 5, norm)"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Initialize RMS weight with ones
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        # RMS normalization: x * rsqrt(mean(x^2) + eps) * weight
        # Compute squared values
        x_sq = x * x
        
        # Compute mean along the last dimension (features)
        mean_sq = x_sq.mean(dim=-1, keepdim=True)
        
        # Compute RMS: sqrt(mean_sq + eps)
        rms = torch.sqrt(mean_sq + self.eps)
        
        # Normalize: x / rms
        normalized = x / rms
        
        # Apply weight
        output = normalized * self.weight
        
        return output


def get_inputs():
    """Generate input tensor for forward pass"""
    # Return a list with one tensor
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    return [input_tensor]


def get_init_inputs():
    """Generate arguments for __init__"""
    # Return a list with hidden_size and eps parameters
    return [NUM_FEATURES, 1e-6]