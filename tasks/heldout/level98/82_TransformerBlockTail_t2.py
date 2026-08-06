import torch
import torch.nn as nn

"""
TransformerBlockTail (tier 2, elementwise)
"""

# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_DIM = 8

class Model(nn.Module):
    """TransformerBlockTail (tier 2, elementwise)"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward layers (small MLP tail)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Layer normalization
        x_norm = self.norm(x)
        
        # Residual connection
        residual = x_norm
        
        # Feed-forward with GELU activation
        hidden = self.fc1(x_norm)
        hidden = self.activation(hidden)
        hidden = self.fc2(hidden)
        
        # Final residual connection
        output = residual + hidden
        return output


def get_inputs():
    """Create input tensor for forward pass"""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]


def get_init_inputs():
    """Create initialization parameters for model"""
    return [HIDDEN_DIM]