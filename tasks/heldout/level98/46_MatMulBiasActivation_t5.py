import torch
import torch.nn as nn

class Model(nn.Module):
    """MatMulBiasActivation (tier 5, matmul)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias with fixed values for determinism
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Use a simple activation
        self.relu = nn.ReLU()
        
        # Ensure deterministic behavior
        self.eval()

    def forward(self, x):
        # Matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
        out = torch.matmul(x, self.weight.t())
        # Add bias: (batch_size, out_features) + (out_features,)
        out = out + self.bias
        # Apply activation
        out = self.relu(out)
        return out

# Define tensor shapes
INPUT_FEATURES = 128
OUTPUT_FEATURES = 128
BATCH_SIZE = 32

def get_inputs():
    # Create input tensor with fixed values for determinism
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    return [INPUT_FEATURES, OUTPUT_FEATURES]