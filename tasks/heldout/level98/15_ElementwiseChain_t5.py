import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        # Register a buffer for deterministic computation
        self.register_buffer('_scale', torch.tensor(1.5))
        self.register_buffer('_offset', torch.tensor(2.0))

    def forward(self, x):
        # Chain of elementwise operations on input tensor x
        # 1. Multiply by scale (elementwise)
        x = x * self._scale
        # 2. Add offset (elementwise)
        x = x + self._offset
        # 3. Square (elementwise)
        x = x * x
        # 4. ReLU activation (elementwise)
        x = torch.relu(x)
        # 5. Tanh activation (elementwise)
        x = torch.tanh(x)
        return x


# Module-level constants for shape configuration
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNELS = 32

def get_inputs():
    # Generate input tensor with fixed values for deterministic results
    # Size: (batch_size=1, channels, height, width)
    input_tensor = torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    # No initialization arguments needed for this model
    return []