import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, pool)"""

    def __init__(self, kernel_size=3, stride=2):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Average pooling with the configured kernel size and stride
        pooled = nn.functional.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Element-wise operation: add a small constant (deterministic)
        result = pooled + 0.1
        
        return result

# Module-level constants for shape configuration
BATCH_SIZE = 3
IN_CHANNELS = 16
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
def get_inputs():
    """Return a list of tensors to pass to forward."""
    # Create input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return []