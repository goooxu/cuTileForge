import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv.eval()  # Set to eval mode for deterministic behavior
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.sigmoid(out)
        return out

# Module-level constants for tensor shapes
INPUT_BATCH_SIZE = 1
INPUT_CHANNELS = 2
INPUT_HEIGHT = 6
INPUT_WIDTH = 6
KERNEL_SIZE = 2
OUT_CHANNELS = 3

def get_inputs():
    # Create input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    # No inputs needed for initialization
    return []