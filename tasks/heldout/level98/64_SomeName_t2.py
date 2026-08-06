import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 4
IN_FEATURES = 8
HIDDEN_FEATURES = 16
OUT_FEATURES = 8

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        # First convolution layer
        self.conv1 = nn.Conv1d(in_features, hidden_features, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_features)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(hidden_features, out_features, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_features)
        
        # Residual projection
        self.residual_proj = nn.Conv1d(in_features, out_features, 1)
        
        # Activation
        self.activation = nn.GELU()
        
        # Set normalization layers to eval mode for deterministic behavior
        self.norm1.eval()
        self.norm2.eval()
        
    def forward(self, x):
        # x shape: (batch, in_features, seq_len)
        batch_size, in_features, seq_len = x.shape
        
        # Residual path
        residual = self.residual_proj(x)
        
        # Main path
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        # Residual connection
        out = x + residual
        
        return out


def get_inputs():
    """Returns input tensors for the model"""
    return [
        torch.randn(BATCH_SIZE, IN_FEATURES, SEQ_LEN)
    ]


def get_init_inputs():
    """Returns initialization arguments for the model"""
    return [IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES]