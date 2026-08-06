import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 3, norm)"""
    
    def __init__(self, dim=32, eps=1e-6):
        super(Model, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Compute RMS normalization
        # x: (batch_size, seq_len, dim)
        # rms = sqrt(mean(x^2, dim=-1, keepdim=True))
        # output = x / (rms + eps) * weight
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = (variance + self.eps).sqrt()
        x = x / rms
        return x * self.weight


# Module-level constants for shapes
BATCH_SIZE = 8
SEQ_LEN = 16
DIM = 32

def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, DIM)]

def get_init_inputs():
    """Return arguments for __init__"""
    return [DIM, 1e-6]