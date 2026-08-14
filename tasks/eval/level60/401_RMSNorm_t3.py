import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 3, norm)"""

    def __init__(self, hidden_size=4096, eps=1e-6, batch_size=32, seq_length=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Create learnable parameter for scaling
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        # RMSNorm computation: x / sqrt(mean(x^2) + eps) * weight
        # Keep original dtype for computation
        original_dtype = x.dtype
        
        # Compute RMS: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        normalized = (x.float() / rms) * self.weight.float()
        
        return normalized.to(original_dtype)


# Module-level constants for tensor shapes
HIDDEN_SIZE = 4096
BATCH_SIZE = 48
SEQ_LENGTH = 2048

def get_inputs():
    """Generate input tensor for RMSNorm"""
    return [torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [HIDDEN_SIZE]