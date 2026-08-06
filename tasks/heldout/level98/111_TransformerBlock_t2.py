import torch
import torch.nn as nn

"""TransformerBlock (tier 2, conv)"""

class Model(nn.Module):
    """TransformerBlock (tier 2, conv)"""
    
    def __init__(self, hidden_dim=512, intermediate_dim=2048):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Layer normalization (norm layer)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # First linear layer (matmul-like operation for intermediate projection)
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        
        # Second linear layer (matmul-like operation for output projection)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        
        # GELU activation
        self.activation = nn.GELU()
        
        # Dropout with 0.0 for deterministic behavior
        self.dropout = nn.Dropout(0.0)
        
        # BatchNorm for normalization (tier 2 category: conv-like)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Set to eval mode for deterministic behavior
        self.batch_norm.eval()

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # First linear projection (matmul + bias)
        x = self.linear1(x)
        
        # GELU activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Second linear projection (matmul + bias)
        x = self.linear2(x)
        
        # Add residual connection
        x = x + residual
        
        # Apply batch norm (treating as a normalisation step in the transformer block)
        # Reshape for batch norm: (batch * seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        x_reshaped = x.reshape(-1, hidden_dim)
        x_reshaped = self.batch_norm(x_reshaped)
        x = x_reshaped.reshape(batch_size, seq_len, hidden_dim)
        
        return x

# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 32
HIDDEN_DIM = 512
INTERMEDIATE_DIM = 2048

def get_inputs():
    """Return a list of tensors to pass to forward()"""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Return a list of arguments to pass to __init__"""
    return [HIDDEN_DIM, INTERMEDIATE_DIM]