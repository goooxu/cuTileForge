import torch
import torch.nn as nn

"""SomeName (tier 3, reduction)"""

# Module-level constants for shapes
REDUCTION_DIM = 1
INPUT_CHANNELS = 8
REDUCED_SIZE = 4
OUTPUT_FEATURES = 3
BATCH_SIZE = 2

class Model(nn.Module):
    """SomeName (tier 3, reduction)"""

    def __init__(self):
        super().__init__()
        # Initialize modules
        self.linear = nn.Linear(REDUCED_SIZE, OUTPUT_FEATURES)
        # Set the module to evaluation mode for deterministic behavior
        self.linear.eval()

    def forward(self, x):
        # x is expected to have shape (batch, channels, ...) where channels=INPUT_CHANNELS
        # Reduction along the channel axis
        # x shape: (2, 8, 2, 2) example
        # After sum along dim=1: (2, 2, 2)
        reduced = torch.sum(x, dim=REDUCTION_DIM, keepdim=False)
        
        # Flatten for the linear layer
        # reduced shape: (2, 2, 2) -> flatten to (2, 4)
        flattened = reduced.view(x.shape[0], -1)
        
        # Apply linear transformation
        # Output shape: (2, 3)
        output = self.linear(flattened)
        
        return output

def get_inputs():
    # Create input tensor of shape (BATCH_SIZE, INPUT_CHANNELS, 2, 2)
    # Total elements: 2 * 8 * 2 * 2 = 64
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, 2, 2)]

def get_init_inputs():
    # No configuration needed, so return an empty list
    return []