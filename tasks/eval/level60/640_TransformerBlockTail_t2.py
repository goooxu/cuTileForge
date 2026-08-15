import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlockTail (tier 2, elementwise)"""
    
    def __init__(self, hidden_dim=64, intermediate_dim=256):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Layer normalization (elementwise operation)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network with linear layers
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        # Residual connection: store original input
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Feed-forward network
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        # Add residual connection
        x = x + residual
        
        return x


# Module-level constants for shapes
HIDDEN_DIM = 65
INTERMEDIATE_DIM = 256
BATCH_SIZE = 7
SEQ_LENGTH = 32

def get_inputs():
    """Return input tensors for the forward pass"""
    # Create a tensor of shape (batch_size, seq_length, hidden_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Return arguments for __init__"""
    return [HIDDEN_DIM, INTERMEDIATE_DIM]