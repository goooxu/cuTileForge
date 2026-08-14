import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 8
KERNEL_SIZE = 3
BATCH_SIZE = 2
HEIGHT = 8
WIDTH = 8

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE]