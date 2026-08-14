import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionModel (tier 5, reduction)"""

    def __init__(self, input_dim=256, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create a linear layer for post-reduction elementwise work
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x has shape [batch_size, input_dim]
        # Reduction along axis 1 (sum)
        reduced = torch.sum(x, dim=1, keepdim=True)
        
        # Elementwise multiplication
        multiplied = reduced * x
        
        # Another reduction along axis 1 (mean)
        final_reduced = torch.mean(multiplied, dim=1)
        
        # Add bias and apply linear transformation
        result = self.linear(x)
        
        # Combine with reduced value
        output = result + final_reduced.unsqueeze(-1)
        
        return output


INPUT_DIM = 256
OUTPUT_DIM = 128
BATCH_SIZE = 96
def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_DIM)]

def get_init_inputs():
    return [INPUT_DIM, OUTPUT_DIM]