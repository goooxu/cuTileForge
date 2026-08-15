import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNormLayer (tier 2, norm)"""

    def __init__(self, normalized_shape, eps=1e-4):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        # Compute RMS normalization
        # x: input tensor, weight: learned scaling parameter
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        normalized = x / rms * self.weight
        
        return normalized

# Module-level constants for tensor shapes
BATCH_SIZE = 6
SEQ_LEN = 64
HIDDEN_DIM = 129
def get_inputs():
    """Returns a list of input tensors for the forward pass."""
    # Create input tensor with shape (batch_size, seq_len, hidden_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Returns a list of arguments for the __init__ method."""
    return [HIDDEN_DIM]