import torch
import torch.nn as nn

"""ConvReLUConv (tier 5, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 8
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
INPUT_HEIGHT = 128
INPUT_WIDTH = 128

class Model(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=1)
        self.relu2 = nn.ReLU()
        
        # For deterministic behavior in eval mode
        self.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


def get_inputs():
    # Create input tensor with fixed values for reproducibility
    # Shape: (batch_size, in_channels, height, width)
    input_shape = (BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    inputs = [torch.randn(input_shape, dtype=torch.float32)]
    return inputs


def get_init_inputs():
    # Return the arguments to pass to __init__
    return [IN_CHANNELS, OUT_CHANNELS]