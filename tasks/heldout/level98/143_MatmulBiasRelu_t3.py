import torch
import torch.nn as nn

"""MatmulBiasRelu (tier 3, matmul)"""

# Module-level constants for tensor shapes
INPUT_FEATURES = 256
OUTPUT_FEATURES = 512
BATCH_SIZE = 64
SEQ_LEN = 32

class Model(nn.Module):
    """SomeName (tier 3, matmul)"""
    
    def __init__(self, input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES):
        super(Model, self).__init__()
        # Define the matrix multiplication layer
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        # Define bias
        self.bias = nn.Parameter(torch.empty(output_features))
        # Initialize weights and bias
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias with deterministic values"""
        # Use kaiming initialization with nonlinearity=None for determinism
        nn.init.kaiming_uniform_(self.weight, a=1.0, mode='fan_in', nonlinearity='linear')
        # Initialize bias with zeros for determinism
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """Forward pass: matrix multiply, add bias, apply ReLU"""
        # Matrix multiplication: (batch, seq, input_features) @ (input_features, output_features)
        # Result shape: (batch, seq, output_features)
        out = torch.matmul(x, self.weight.t())
        # Add bias: broadcasting over batch and seq dimensions
        out = out + self.bias
        # Apply ReLU activation
        out = torch.relu(out)
        return out

def get_inputs():
    """Generate deterministic input tensors"""
    # Create input tensor with shape (BATCH_SIZE, SEQ_LEN, INPUT_FEATURES)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_FEATURES)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [INPUT_FEATURES, OUTPUT_FEATURES]