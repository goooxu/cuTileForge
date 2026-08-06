import torch
import torch.nn as nn

"""SomeName (tier 3, matmul)"""


class Model(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, has_bias=True):
        super(Model, self).__init__()
        self.has_bias = has_bias
        
        # Linear layer for matrix multiplication with optional bias
        self.linear = nn.Linear(input_features, hidden_features, bias=has_bias)
        self.linear.eval()  # Ensure deterministic behavior
        
        # Additional linear layer for second matrix multiply with bias
        self.linear2 = nn.Linear(hidden_features, output_features, bias=has_bias)
        self.linear2.eval()
        
        # Activation function
        self.activation = nn.ReLU()
        self.activation.eval()

    def forward(self, x1, x2=None):
        # If only one input is provided, use it directly
        if x2 is None:
            x = x1
        else:
            # Perform matrix multiplication on the first two inputs
            x = torch.matmul(x1, x2)
        
        # Apply first linear transformation
        out = self.linear(x)
        
        # Apply activation
        out = self.activation(out)
        
        # Apply second linear transformation
        out = self.linear2(out)
        
        return out


# Module-level constants for shapes
INPUT_FEATURES = 8
HIDDEN_FEATURES = 16
OUTPUT_FEATURES = 4
BATCH_SIZE = 4

def get_inputs():
    # Generate two small tensors for matrix multiplication
    # Shape: (BATCH_SIZE, INPUT_FEATURES) and (INPUT_FEATURES, HIDDEN_FEATURES)
    # For simplicity, we'll generate a single input tensor and let the model handle it
    # The model can work with either one input (uses self.linear directly) or two inputs (uses torch.matmul)
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    # Return the arguments for __init__
    return [INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES]