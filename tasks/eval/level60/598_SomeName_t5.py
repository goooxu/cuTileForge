import torch
import torch.nn as nn

# Module-level constants for shape definitions
BATCH_SIZE = 6
IN_FEATURES = 8
OUT_FEATURES = 6

class Model(nn.Module):
    """SomeName (tier 5, matmul)"""
    
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight and bias for matrix multiplication
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # Matrix multiplication: (BATCH_SIZE, in_features) @ (in_features, out_features) -> (BATCH_SIZE, out_features)
        result = torch.matmul(x, self.weight)
        # Add bias: (BATCH_SIZE, out_features) + (out_features,) -> (BATCH_SIZE, out_features)
        result = result + self.bias
        # Apply ReLU activation
        result = torch.relu(result)
        return result

def get_inputs():
    # Return input tensor with shape (BATCH_SIZE, IN_FEATURES)
    return [torch.randn(BATCH_SIZE, IN_FEATURES)]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_FEATURES, OUT_FEATURES]