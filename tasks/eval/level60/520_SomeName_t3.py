import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 13
SEQ_LEN = 513
HIDDEN_DIM = 1025
class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, hidden_dim=HIDDEN_DIM, kernel_size=3, padding=1):
        super(Model, self).__init__()
        
        # Convolutional layer for feature transformation
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Set to eval mode for deterministic behavior
        self.norm.eval()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # Residual connection
        residual = x
        
        # Permute to (batch_size, hidden_dim, seq_len) for conv1d
        x = x.permute(0, 2, 1)
        
        # Apply convolution
        x = self.conv(x)
        
        # Permute back to (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)
        
        # Add residual and normalize
        x = self.norm(x + residual)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        
        # Final residual connection
        output = x + ffn_out
        
        return output

def get_inputs():
    """Return input tensors for the model."""
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    ]

def get_init_inputs():
    """Return initialization arguments for the model."""
    return [HIDDEN_DIM]