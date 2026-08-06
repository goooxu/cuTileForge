import torch
import torch.nn as nn

"""SomeName (tier 5, pool)"""

# Module-level constants for shapes
BATCH_SIZE = 4
IN_CHANNELS = 16
H = 64
W = 64

# Input size
INPUT_SIZE = (BATCH_SIZE, IN_CHANNELS, H, W)

# Output size after average pooling with kernel_size=2, stride=2
# (H_out = H // 2 = 32, W_out = W // 2 = 32)
OUT_CHANNELS = IN_CHANNELS
OUT_H = H // 2
OUT_W = W // 2
OUTPUT_SIZE = (BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W)


class Model(nn.Module):
    """SomeName (tier 5, pool)"""

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Define elementwise operation weights as parameters
        self.elementwise_weight = nn.Parameter(torch.ones(OUT_CHANNELS))
        self.elementwise_bias = nn.Parameter(torch.zeros(OUT_CHANNELS))

    def forward(self, x):
        # Apply pooling layer
        pooled = self.pool(x)
        
        # Apply elementwise multiplication and addition
        # Reshape weight/bias for broadcasting: (B, C, H, W) with C being OUT_CHANNELS
        result = pooled * self.elementwise_weight.view(1, -1, 1, 1) + self.elementwise_bias.view(1, -1, 1, 1)
        
        return result


def get_inputs():
    """Return input tensor for the model."""
    # Create input tensor with deterministic values
    input_tensor = torch.randn(INPUT_SIZE, requires_grad=True)
    return [input_tensor]


def get_init_inputs():
    """Return initialization arguments for the model."""
    return []