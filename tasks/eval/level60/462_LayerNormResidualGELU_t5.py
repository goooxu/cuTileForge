import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormResidualGELU (tier 5, elementwise)"""

    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.eval()  # Ensure deterministic behavior

    def forward(self, x, residual):
        # Normalize the input
        x = self.layer_norm(x)
        # Add residual
        x = x + residual
        # Apply activation
        x = self.activation(x)
        return x


# Module-level constants for shape configuration
HIDDEN_SIZE = 1024
BATCH_SIZE = 6
SEQ_LENGTH = 64

def get_inputs():
    """Return list of tensors for forward pass"""
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    residual = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x, residual]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [HIDDEN_SIZE]