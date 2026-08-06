import torch
import torch.nn as nn

class Model(nn.Module):
    """MatmulReLU (tier 5, matmul)"""
    
    def __init__(self, in_features, hidden_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        # Create learnable parameters
        self.weight = nn.Parameter(torch.empty(in_features, hidden_features))
        self.bias = nn.Parameter(torch.empty(hidden_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Register buffers for deterministic behavior
        self.register_buffer('zeros', torch.zeros(1))
    
    def forward(self, input_tensor):
        # Matrix multiplication: (N, in_features) @ (in_features, hidden_features) -> (N, hidden_features)
        output = torch.matmul(input_tensor, self.weight)
        
        # Add bias: (N, hidden_features) + (hidden_features,) -> (N, hidden_features)
        output = output + self.bias
        
        # Apply ReLU activation: element-wise operation
        output = torch.relu(output)
        
        return output


# Module-level constants for shape configuration
INPUT_BATCH_SIZE = 4
INPUT_FEATURES = 8
HIDDEN_FEATURES = 12
OUTPUT_FEATURES = 6

def get_inputs():
    """Returns a list of tensors to pass to forward method"""
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES]

# Import math for sqrt and other mathematical functions
import math