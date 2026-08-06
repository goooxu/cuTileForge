import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

# Module-level constants for shapes
INPUT_FEATURES = 8
INNER_FEATURES = 16
OUTPUT_FEATURES = 12
BATCH_SIZE = 4

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # We'll create learned parameters for the operations
        self.weight1 = nn.Parameter(torch.randn(INPUT_FEATURES, INNER_FEATURES))
        self.weight2 = nn.Parameter(torch.randn(INNER_FEATURES, OUTPUT_FEATURES))
        self.bias1 = nn.Parameter(torch.randn(INNER_FEATURES))
        self.bias2 = nn.Parameter(torch.randn(OUTPUT_FEATURES))
        self.register_buffer('scale', torch.tensor(1.5))
        self.register_buffer('shift', torch.tensor(0.5))
        
    def forward(self, x):
        # Chain of elementwise operations:
        # 1. Linear transformation: x @ weight1 + bias1
        # 2. Elementwise activation: tanh
        # 3. Scale and shift operations
        # 4. Linear transformation to output: result @ weight2 + bias2
        
        # Step 1: First linear transform (batch_size, input_features) @ (input_features, inner_features)
        x = torch.matmul(x, self.weight1) + self.bias1
        
        # Step 2: Elementwise activation
        x = torch.tanh(x)
        
        # Step 3: Elementwise scaling and shifting (scale and shift are broadcastable)
        x = x * self.scale + self.shift
        
        # Step 4: Second linear transform to output features
        x = torch.matmul(x, self.weight2) + self.bias2
        
        return x

def get_inputs():
    """Return a list of input tensors."""
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    """Return arguments for __init__."""
    return []