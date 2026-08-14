import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

# Module-level constants for shapes
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 16
BATCH_SIZE = 3
SEQ_LEN = 4

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Convolution layers for transformer-style block
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Activation function
        self.act = nn.ReLU()
        
        # Set evaluation mode for deterministic behavior
        self.norm1.eval()
        self.norm2.eval()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Residual connection
        residual = x
        
        # Layer normalization
        x = self.norm1(x)
        
        # Transpose for 1D convolution: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # First conv layer
        x = self.conv1(x)
        
        # Activation
        x = self.act(x)
        
        # Second conv layer
        x = self.conv2(x)
        
        # Transpose back: (batch_size, output_dim, seq_len) -> (batch_size, seq_len, output_dim)
        x = x.permute(0, 2, 1)
        
        # Add residual connection
        x = x + residual
        
        # Second normalization
        x = self.norm2(x)
        
        return x

def get_inputs():
    """Generate input tensor for the model."""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)]

def get_init_inputs():
    """Generate arguments for model initialization."""
    return [INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM]