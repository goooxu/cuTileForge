import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

class Model(nn.Module):
    """SomeName (tier 2, matmul)"""

    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
        
        # No BatchNorm, but if we had one, we'd call .eval() here
        self.activation = nn.ReLU()

    def reset_parameters(self):
        # Kaiming initialization for matmul-like operations
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Matrix multiplication
        out = torch.matmul(x, self.weight.t())
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        # Apply activation
        out = self.activation(out)
        return out


# Module-level constants for shapes
IN_FEATURES = 256
OUT_FEATURES = 512
BATCH_SIZE = 32

def get_inputs():
    """Return list of tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_FEATURES)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [IN_FEATURES, OUT_FEATURES, True]