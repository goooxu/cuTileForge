import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 16
INPUT_SIZE = 64  # 1D tensor size for each sample
OUTPUT_SIZE = INPUT_SIZE  # Elementwise operations preserve size

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # No learnable parameters needed for pure elementwise ops
        # but we need to store configuration for reproducibility
        self.config = config
    
    def forward(self, x):
        # Chain of 4+ elementwise operations
        # Operation 1: Softplus activation
        y1 = torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)
        
        # Operation 2: Sigmoid activation
        y2 = torch.sigmoid(y1)
        
        # Operation 3: Tanh activation
        y3 = torch.tanh(y2)
        
        # Operation 4: ReLU activation
        y4 = torch.relu(y3)
        
        # Operation 5: Absolute value
        y5 = torch.abs(y4)
        
        # Operation 6: Square
        y6 = y5 * y5
        
        # Operation 7: Sqrt (to ensure differentiability and avoid NaN)
        y7 = torch.sqrt(y6 + 1e-8)
        
        return y7

def get_inputs():
    """Generate a list of input tensors for forward pass."""
    # Create input tensor of appropriate size
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, OUTPUT_SIZE)
    return [x]

def get_init_inputs():
    """Generate arguments to pass to __init__."""
    return [{}]  # Empty config dict