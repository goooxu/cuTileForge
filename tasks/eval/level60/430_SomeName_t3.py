import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 6
IN_FEATURES = 8
OUT_FEATURES = 6

class Model(nn.Module):
    """SomeName (tier 3, matmul)"""

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear layer combines matrix multiply, bias, and can include activation
        # We'll use a linear layer which does: y = x @ W^T + b
        self.linear = nn.Linear(in_features, out_features)
        
        # Add activation as a separate module
        self.activation = nn.ReLU()

    def forward(self, x):
        # Matrix multiply (via linear layer) + bias + activation
        x = self.linear(x)
        x = self.activation(x)
        return x

def get_inputs():
    # Create input tensor of shape (BATCH_SIZE, IN_FEATURES)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    return [x]

def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES]