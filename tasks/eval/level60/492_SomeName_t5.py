import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 12
IN_FEATURES = 64
OUT_FEATURES = 128

class Model(nn.Module):
    """SomeName (tier 5, matmul)"""
    
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # Matrix multiplication followed by bias and activation
        x = torch.matmul(x, self.weight.t())
        x = x + self.bias
        x = torch.relu(x)
        return x

def get_inputs():
    # Return input tensor for forward pass
    return [torch.randn(BATCH_SIZE, IN_FEATURES)]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_FEATURES, OUT_FEATURES]