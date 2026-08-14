import torch
import torch.nn as nn

# Module-level constants for shapes
INPUT_DIM = 64
HIDDEN_DIM = 256
BATCH_SIZE = 16
SEQ_LENGTH = 32

class Model(nn.Module):
    """TransformerBlock (tier 3, elementwise)"""
    
    def __init__(self, input_dim, hidden_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # Feed-forward layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
        # Activation function
        self.gelu = nn.GELU()
        
        # Dropout is not used to ensure deterministic behavior
        
    def forward(self, x):
        # Store residual for residual connection
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # First feed-forward layer with GELU activation
        x = self.fc1(x)
        x = self.gelu(x)
        
        # Second feed-forward layer
        x = self.fc2(x)
        
        # Add residual connection
        x = x + residual
        
        return x

def get_inputs():
    # Create input tensor with shape [batch_size, seq_length, input_dim]
    return [torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)]

def get_init_inputs():
    # Return configuration arguments for model initialization
    return [INPUT_DIM, HIDDEN_DIM]