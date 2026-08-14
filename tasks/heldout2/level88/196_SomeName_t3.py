import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, input_size, batch_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_norm.eval()  # Make deterministic

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = torch.sigmoid(x)
        return x


# Module-level constants for shape configuration
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
INPUT_SIZE = 512
BATCH_SIZE = 32


def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_SIZE, BATCH_SIZE]