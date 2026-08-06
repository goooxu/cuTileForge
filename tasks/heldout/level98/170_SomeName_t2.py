import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""
    
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.in_features = in_features
        
        # Define layer - this will be followed by 4+ elementwise operations
        self.layer = nn.Linear(in_features, in_features)
        
        # Initialize weights to be deterministic
        nn.init.constant_(self.layer.weight, 0.01)
        nn.init.constant_(self.layer.bias, 0.0)
        
        # Ensure module is in eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        # Chain of elementwise operations on the tensor
        out = self.layer(x)          # Linear transformation
        out = torch.relu(out)        # ReLU activation (elementwise)
        out = torch.sigmoid(out)     # Sigmoid (elementwise)
        out = out * 0.5              # Elementwise multiplication
        out = out + 0.1              # Elementwise addition
        return out


# Module-level constants for shapes
INPUT_FEATURES = 32

def get_inputs():
    """Returns input tensors for the forward pass."""
    # Create input tensor of appropriate size
    # Input shape: (batch_size, in_features)
    batch_size = 8
    x = torch.randn(batch_size, INPUT_FEATURES)
    return [x]

def get_init_inputs():
    """Returns arguments for __init__."""
    return [INPUT_FEATURES]