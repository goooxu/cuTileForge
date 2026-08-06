import torch
import torch.nn as nn

"""SomeName (reduction)"""


class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    def __init__(self, input_dim=16, output_dim=8, reduction_dim=3):
        super(Model, self).__init__()
        self.reduction_dim = reduction_dim
        # This will be used for the elementwise operation after reduction
        self.scale = nn.Parameter(torch.ones(input_dim))
        # Note: Using nn.Linear for elementwise transform after reduction
        self.transform = nn.Linear(1, output_dim, bias=False)
        # Ensure deterministic behavior
        for param in self.parameters():
            torch.nn.init.constant_(param, 0.1)
        # For evaluation mode to ensure deterministic BatchNorm behavior if used
        # But we don't use BatchNorm in this implementation, so this line is a no-op
        self.eval()

    def forward(self, x):
        # x shape: [batch, input_dim, reduction_dim]
        # First, perform reduction along the reduction_dim (axis=2)
        reduced = torch.sum(x, dim=2)  # [batch, input_dim]
        
        # Elementwise: scale by learnable parameter
        scaled = reduced * self.scale  # [batch, input_dim]
        
        # Reshape for transform: add a dimension for the linear layer
        # scaled shape is [batch, input_dim], we need [batch, input_dim, 1]
        scaled = scaled.unsqueeze(-1)  # [batch, input_dim, 1]
        
        # Apply elementwise linear transformation (each feature gets transformed independently)
        result = self.transform(scaled)  # [batch, input_dim, output_dim]
        
        # Reshape to final output: [batch, input_dim * output_dim]
        result = result.view(result.size(0), -1)
        
        return result


# Module-level constants for shapes
INPUT_DIM = 16
OUTPUT_DIM = 8
REDUCTION_DIM = 3
BATCH_SIZE = 4

def get_inputs():
    """Returns a list of tensors to pass to forward method."""
    # Create input tensor with shape [batch_size, input_dim, reduction_dim]
    input_tensor = torch.ones(BATCH_SIZE, INPUT_DIM, REDUCTION_DIM)
    return [input_tensor]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [INPUT_DIM, OUTPUT_DIM, REDUCTION_DIM]