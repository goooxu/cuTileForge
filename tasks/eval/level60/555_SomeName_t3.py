import torch
import torch.nn as nn

"""SomeName (tier 3, matmul)"""

# Module-level constants for shapes
INPUT_FEATURES = 256
OUTPUT_FEATURES = 512
BATCH_SIZE = 48
class Model(nn.Module):
    """SomeName (tier 3, matmul)"""
    
    def __init__(self, input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES):
        super(Model, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # Linear layer performs matrix multiplication + bias
        self.linear = nn.Linear(input_features, output_features)
        
        # Use ReLU activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Matrix multiplication (via linear layer) + bias + activation
        x = self.linear(x)
        x = self.activation(x)
        return x

def get_inputs():
    # Create input tensor with appropriate shape
    # For batched matrix multiplication: (batch_size, input_features)
    input_tensor = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    return [input_tensor]

def get_init_inputs():
    # Return configuration arguments for __init__
    return [INPUT_FEATURES, OUTPUT_FEATURES]