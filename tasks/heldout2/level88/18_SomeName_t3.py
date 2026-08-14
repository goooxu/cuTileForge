import torch
import torch.nn as nn

# Module-level constants for shapes
INPUT_SIZE = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 128

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""
    
    def __init__(self, in_features=INPUT_SIZE, hidden_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        # Linear layers to project to required sizes
        self.proj1 = nn.Linear(in_features, hidden_features)
        self.proj2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        # Chain of elementwise operations
        x = self.proj1(x)
        x = x * 2.0  # elementwise multiply
        x = x + 1.0  # elementwise add
        x = torch.relu(x)  # elementwise activation
        x = x * x  # elementwise square
        x = self.proj2(x)
        return x

def get_inputs():
    # Create input tensor with shape matching expected input
    # Using batch size of 1 for simplicity, can be adjusted
    return [torch.randn(1, INPUT_SIZE)]

def get_init_inputs():
    return []