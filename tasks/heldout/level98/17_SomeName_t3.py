import torch
import torch.nn as nn

"""PoolMax2DRelu (pool, elementwise)"""

# Module-level constants for shape configuration
BATCH_SIZE = 2
INPUT_CHANNELS = 3
INPUT_HEIGHT = 4
INPUT_WIDTH = 4
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0

class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Max pooling layer with specified parameters
        self.pool = nn.MaxPool2d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            return_indices=False
        )
        # ReLU activation for elementwise operation
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Apply max pooling
        x = self.pool(x)
        # Apply ReLU activation (elementwise operation)
        x = self.relu(x)
        return x

def get_inputs():
    # Create input tensor with deterministic values
    # Shape: (batch_size, channels, height, width) = (2, 3, 4, 4)
    input_tensor = torch.arange(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH, dtype=torch.float32)
    input_tensor = input_tensor.view(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    # No initialization arguments needed for this model
    return []