import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.eval()  # Set to eval mode to ensure deterministic behavior

    def forward(self, x):
        pooled = self.pool(x)
        # Elementwise operation: apply softsign function
        result = torch.nn.functional.softsign(pooled)
        return result

# Module-level constants for shape configuration
INPUT_BATCH = 1
INPUT_CHANNELS = 256
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
KERNEL_SIZE = 4
STRIDE = 4
PADDING = 0

def get_inputs():
    # Create input tensor with appropriate dimensions
    return [torch.empty(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).uniform_(-1.0, 1.0)]

def get_init_inputs():
    # Return initialization arguments that match the expected parameters
    return [KERNEL_SIZE, STRIDE, PADDING]