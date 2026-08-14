import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 2, norm)"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Create learnable scale parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x):
        # Compute RMS normalization
        # x: [batch_size, seq_len, hidden_size]
        # We normalize along the hidden dimension
        
        # Compute square mean along hidden dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # Normalize and scale
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        
        return x


# Shape constants for large tensor testing
BATCH_SIZE = 32
SEQ_LEN = 1024
HIDDEN_SIZE = 4096

def get_inputs():
    """Return list of input tensors"""
    # Create a tensor that matches the expected input shape
    # [batch_size, seq_len, hidden_size]
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [HIDDEN_SIZE]