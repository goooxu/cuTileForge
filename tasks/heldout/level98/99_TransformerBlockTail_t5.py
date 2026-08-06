import torch
import torch.nn as nn

"""TransformerBlockTail (tier 5, norm)"""
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        
        # Feed-forward layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        
        # Activation
        self.gelu = nn.GELU()
        
        # Dropout (no randomness if dropout=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # First normalization + residual path
        x_norm = self.norm1(x)
        # Feed-forward with GELU activation
        x_ff = self.fc1(x_norm)
        x_ff = self.gelu(x_ff)
        x_ff = self.fc2(x_ff)
        x_ff = self.dropout(x_ff)
        # Residual connection
        x_out = x + x_ff
        
        # Second normalization for consistency
        x_out = self.norm2(x_out)
        
        return x_out


# Module-level constants for shape configuration
IN_FEATURES = 512
HIDDEN_FEATURES = 2048
DROPOUT = 0.0
BATCH_SIZE = 16
SEQ_LEN = 128

def get_inputs():
    """Return input tensors matching the forward signature"""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, IN_FEATURES)]

def get_init_inputs():
    """Return initialization arguments for Model"""
    return [IN_FEATURES, HIDDEN_FEATURES, DROPOUT]