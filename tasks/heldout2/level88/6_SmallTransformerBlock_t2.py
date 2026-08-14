import torch
import torch.nn as nn

class Model(nn.Module):
    """SmallTransformerBlock (tier 2, elementwise)"""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network with two linear layers
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Set to eval mode for deterministic behavior
        self.norm.eval()

    def forward(self, x):
        # Residual connection
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
HIDDEN_SIZE = 64
BATCH_SIZE = 2
SEQ_LENGTH = 8

def get_inputs():
    """Return a list of tensors to pass to forward"""
    # Create input tensor with shape (batch_size, seq_length, hidden_size)
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x]

def get_init_inputs():
    """Return a list of arguments to pass to __init__"""
    return [HIDDEN_SIZE]