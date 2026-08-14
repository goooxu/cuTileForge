import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_FEATURES = 512
HIDDEN_FEATURES = 2048
BATCH_SIZE = 48
SEQ_LENGTH = 16

class Model(nn.Module):
    """TransformerBlock (tier 3, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(INPUT_FEATURES)
        
        # Two feed-forward layers (MLP) with GELU activation
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.fc2 = nn.Linear(HIDDEN_FEATURES, INPUT_FEATURES)
        self.gelu = nn.GELU()
        
        # Dropout is excluded per requirements (no randomness)
        
    def forward(self, x):
        # Residual connection: store input for later addition
        residual = x
        
        # Layer normalization
        x = self.norm1(x)
        
        # First feed-forward layer
        x = self.fc1(x)
        x = self.gelu(x)
        
        # Second feed-forward layer
        x = self.fc2(x)
        
        # Add residual connection
        x = x + residual
        
        return x

def get_inputs():
    """Generate input tensor for the transformer block"""
    return [torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_FEATURES)]

def get_init_inputs():
    """Return initialization arguments (empty for this model)"""
    return []