import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

# Module-level constants for shapes
BATCH_SIZE = 384
IN_FEATURES = 4096
OUT_FEATURES = 4096

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)
        
        # Initialize weights and bias to deterministic values
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        
        # Use ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Matrix multiply followed by bias and activation
        x = self.linear(x)
        x = self.relu(x)
        return x

def get_inputs():
    # Return input tensor for forward pass
    return [torch.randn(BATCH_SIZE, IN_FEATURES)]

def get_init_inputs():
    # No additional inputs needed for initialization
    return []