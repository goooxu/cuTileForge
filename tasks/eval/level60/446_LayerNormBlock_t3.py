import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormBlock (tier 3, norm)"""

    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # LayerNorm for normalization
        self.ln = nn.LayerNorm(embed_dim)
        
        # Linear layers for the transformation
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Set BatchNorm to eval mode if present (though we don't use BatchNorm here)
        # This is just to show the pattern as required

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # Residual connection
        residual = x
        
        # Layer normalization
        x = self.ln(x)
        
        # First linear transformation
        x = self.fc1(x)
        
        # Activation
        x = self.relu(x)
        
        # Second linear transformation
        x = self.fc2(x)
        
        # Residual connection
        x = x + residual
        
        return x

# Module-level constants for shapes
BATCH_SIZE = 3
SEQ_LEN = 4
EMBED_DIM = 32
HIDDEN_DIM = 64

def get_inputs():
    """Returns a list of tensors to pass to forward."""
    # Create input tensor with shape (batch_size, seq_len, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [EMBED_DIM, HIDDEN_DIM]