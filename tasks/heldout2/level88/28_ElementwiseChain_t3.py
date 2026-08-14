import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_CHANNELS = 64
BATCH_SIZE = 8

class Model(nn.Module):
    """ElementwiseChain (tier 3, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # No learnable parameters needed for pure elementwise operations
        pass
    
    def forward(self, x):
        # Chain of elementwise operations:
        # 1. Add constant
        x = x + 1.0
        
        # 2. Multiply by scalar
        x = x * 0.5
        
        # 3. Apply ReLU activation
        x = torch.relu(x)
        
        # 4. Square the values
        x = x ** 2
        
        # 5. Apply exponential
        x = torch.exp(x)
        
        # 6. Subtract constant
        x = x - 1.0
        
        # 7. Apply logarithm (safe since x > 0 after exp)
        x = torch.log(x + 1e-10)  # small epsilon for numerical stability
        
        return x

def get_inputs():
    """Return list of input tensors for the model"""
    # Create a tensor with shape suitable for elementwise operations
    input_tensor = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return list of arguments for __init__ (empty in this case)"""
    return []