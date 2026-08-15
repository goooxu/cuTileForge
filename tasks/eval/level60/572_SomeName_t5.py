import torch
import torch.nn as nn

# Module-level constants for shapes
INPUT_DIM = 5
HIDDEN_DIM = 8
OUTPUT_DIM = 6
BATCH_SIZE = 3
class Model(nn.Module):
    """SomeName (tier 5, matmul)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(OUTPUT_DIM, INPUT_DIM))
        self.bias = nn.Parameter(torch.randn(OUTPUT_DIM))
    
    def forward(self, x):
        # Matrix multiply: (BATCH_SIZE, INPUT_DIM) @ (INPUT_DIM, OUTPUT_DIM) -> (BATCH_SIZE, OUTPUT_DIM)
        result = torch.matmul(x, self.weight.t())
        # Add bias: (BATCH_SIZE, OUTPUT_DIM) + (OUTPUT_DIM,) -> (BATCH_SIZE, OUTPUT_DIM)
        result = result + self.bias
        # Apply activation (ReLU)
        result = torch.relu(result)
        return result

def get_inputs():
    # Return list of tensors to pass to forward
    return [torch.randn(BATCH_SIZE, INPUT_DIM)]

def get_init_inputs():
    # Return list of arguments to pass to __init__ (empty in this case)
    return []