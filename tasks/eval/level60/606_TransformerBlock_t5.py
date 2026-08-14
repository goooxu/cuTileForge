import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 5, elementwise)"""
    
    def __init__(self, hidden_size=1024, intermediate_size=2048):
        super().__init__()
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network with GELU activation
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        
        # Register buffers for deterministic behavior
        self.register_buffer('_dummy', torch.zeros(1))
        
    def forward(self, x):
        # Save residual
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # Feed-forward network
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        # Residual connection
        x = x + residual
        
        return x


# Module-level constants for shapes
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2048
BATCH_SIZE = 12
SEQ_LEN = 128

def get_inputs():
    """Return input tensors for the forward pass"""
    # Create a tensor with shape (batch_size, seq_len, hidden_size)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    return [x]

def get_init_inputs():
    """Return arguments for the __init__ method"""
    return [HIDDEN_SIZE, INTERMEDIATE_SIZE]