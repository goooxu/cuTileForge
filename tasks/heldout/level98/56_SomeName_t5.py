import torch
import torch.nn as nn

"""
SomeName (tier 5, matmul)
"""

# Shape constants for medium tensors
INPUT_FEATURES = 256
HIDDEN_FEATURES = 512
OUTPUT_FEATURES = 128
BATCH_SIZE = 32

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, input_features=INPUT_FEATURES, hidden_features=HIDDEN_FEATURES, output_features=OUTPUT_FEATURES):
        super(Model, self).__init__()
        # We use the actual shape values provided to the module
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        
        # Matrix multiplication layer: W @ X + b
        # Using Linear layer but configuring it as matmul + bias
        self.matmul = nn.Linear(
            in_features=input_features, 
            out_features=output_features,
            bias=True
        )
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Ensure x is at least 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Matrix multiplication: (batch_size, input_features) @ (input_features, output_features)
        # This is equivalent to x @ self.matmul.weight.T + self.matmul.bias
        out = self.matmul(x)
        
        # Activation
        out = self.activation(out)
        
        return out

def get_inputs():
    """Return input tensors for the model"""
    # Create a tensor of shape (BATCH_SIZE, INPUT_FEATURES)
    # Using zeros since no randomness is allowed in forward
    input_tensor = torch.zeros(BATCH_SIZE, INPUT_FEATURES)
    return [input_tensor]

def get_init_inputs():
    """Return arguments for the __init__ method"""
    # Return the shape parameters
    return [INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES]