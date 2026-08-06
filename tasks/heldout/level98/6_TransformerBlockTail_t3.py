import torch
import torch.nn as nn

"""
Model for kernel porting exercise - transformer-style block tail
"""

class Model(nn.Module):
    """TransformerBlockTail (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size=3):
        super(Model, self).__init__()
        # Configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # Main layers
        self.norm1 = nn.LayerNorm([in_channels])
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=True)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=True)
        
        # Use BatchNorm for the residual branch (eval mode for determinism)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn.eval()
        
        # Residual connection layers
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        batch_size, channels, seq_len = x.shape
        
        # Normalise along channel dimension
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.norm1(x)       # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        # Residual branch
        residual = self.proj(x)  # (batch, out_channels, seq_len)
        residual = self.bn(residual)
        
        # Main branch: conv1 -> activation -> conv2
        out = self.conv1(x)      # (batch, hidden_dim, seq_len)
        out = self.activation(out)
        out = self.conv2(out)    # (batch, out_channels, seq_len)
        
        # Add residual
        out = out + residual     # (batch, out_channels, seq_len)
        
        return out

# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
HIDDEN_DIM = 256
BATCH_SIZE = 4
SEQ_LEN = 128

def get_inputs():
    """Return list of tensors to pass to forward method"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LEN)]

def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, HIDDEN_DIM]