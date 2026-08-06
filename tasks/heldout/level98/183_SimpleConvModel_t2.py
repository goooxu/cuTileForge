import torch
import torch.nn as nn

"""SimpleConvModel (tier 2, conv)"""

# Module-level constants for shapes
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
KERNEL_SIZE = 3
BATCH_SIZE = 1
HEIGHT = 4
WIDTH = 4
PADDING = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolution layer
        self.conv = nn.Conv2d(INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, padding=PADDING)
        # Activation after convolution
        self.relu = nn.ReLU()
        # Another elementwise operation (LeakyReLU)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Elementwise ReLU
        x = self.relu(x)
        # Elementwise LeakyReLU
        x = self.leaky_relu(x)
        return x

def get_inputs():
    # Generate input tensor with fixed values for reproducibility
    return [torch.ones(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # No initialization arguments needed
    return []