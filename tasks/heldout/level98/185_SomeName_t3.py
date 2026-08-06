import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
SEQ_LEN = 128
HIDDEN_DIM = 256
KERNEL_SIZE = 3
NUM_FEATURES = HIDDEN_DIM

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, hidden_dim, kernel_size, num_features):
        super().__init__()
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=1,
            bias=False
        )
        
        # Output projection (pointwise conv to keep dimension consistent)
        self.proj = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=True
        )
        
        # Activation function (ReLU)
        self.act = nn.ReLU()
        
        # Batch normalization for internal computation (eval mode for determinism)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.bn.eval()
        
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_features = num_features
    
    def forward(self, x):
        # x shape: (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        B, S, H = x.shape
        
        # Residual connection
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # Transpose for Conv1d: (B, H, S)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply activation
        x = self.act(x)
        
        # Apply batch normalization (eval mode ensures determinism)
        x = self.bn(x)
        
        # Apply projection
        x = self.proj(x)
        
        # Transpose back: (B, S, H)
        x = x.transpose(1, 2)
        
        # Add residual
        x = x + residual
        
        return x


def get_inputs():
    """Return list of tensors to pass to forward method."""
    # Create input tensor: (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    ]


def get_init_inputs():
    """Return list of arguments to pass to __init__."""
    return [HIDDEN_DIM, KERNEL_SIZE, NUM_FEATURES]