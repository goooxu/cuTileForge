import torch
import torch.nn as nn

"""
Model for transformer-style block with normalization, residual connection, and activation.
"""

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Linear transformations for residual and activation
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Ensure deterministic behavior
        self.eval()

    def forward(self, x):
        # Store original input for residual connection
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # First linear transformation
        x = self.linear1(x)
        
        # Apply activation function
        x = self.activation(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        # Add residual connection
        x = x + residual
        
        return x


# Module-level constants for shapes
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
BATCH_SIZE = 5
SEQ_LENGTH = 4

def get_inputs():
    """Return input tensors for the model."""
    # Create random input tensor with shape (batch_size, seq_length, hidden_size)
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x]

def get_init_inputs():
    """Return initialization arguments for the model."""
    return [HIDDEN_SIZE, INTERMEDIATE_SIZE]