import torch
import torch.nn as nn

class Model(nn.Module):
    """LayernormResidualGelu (tier 5, norm)"""

    def __init__(self, hidden_size=1024, eps=1e-5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.gelu = nn.GELU()

    def forward(self, x, residual):
        # Apply layernorm to input
        x_norm = self.layernorm(x)
        # Add residual
        x_res = x_norm + residual
        # Apply activation
        output = self.gelu(x_res)
        return output

# Module-level constants for shape configuration
HIDDEN_SIZE = 1024
BATCH_SIZE = 8
SEQ_LENGTH = 512

def get_inputs():
    """Returns list of tensors for forward pass"""
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    residual = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x, residual]

def get_init_inputs():
    """Returns list of arguments for __init__"""
    return [HIDDEN_SIZE, 1e-5]