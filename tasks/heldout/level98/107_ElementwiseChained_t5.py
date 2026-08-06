import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChained (tier 5, elementwise)"""
    
    def __init__(self, input_features):
        super(Model, self).__init__()
        # Placeholder for configuration
        self.input_features = input_features
    
    def forward(self, x):
        # Chain of 5 elementwise operations: relu -> tanh -> relu -> tanh -> relu
        out = torch.relu(x)
        out = torch.tanh(out)
        out = torch.relu(out)
        out = torch.tanh(out)
        out = torch.relu(out)
        return out

# Module-level constants for shapes
BATCH_SIZE = 64
FEATURES = 1024

def get_inputs():
    # Generate input tensor with consistent values for deterministic behavior
    x = torch.ones(BATCH_SIZE, FEATURES) * 0.1  # Small positive values
    return [x]

def get_init_inputs():
    # Return configuration arguments for __init__
    return [FEATURES]