import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, leaky_alpha=0.01):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        
        # Convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        # Batch normalization if enabled
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            self.bn.eval()  # Ensure deterministic behavior
        
        # Additional elementwise operations: ReLU and a scaling operation
        self.leaky_alpha = leaky_alpha

    def forward(self, x):
        # Convolution
        out = self.conv(x)
        
        # Batch normalization (eval mode ensures deterministic)
        if self.batch_norm:
            out = self.bn(out)
        
        # Leaky ReLU (elementwise operation 1)
        out = torch.where(out > 0, out, self.leaky_alpha * out)
        
        # Elementwise multiplication by a constant factor (elementwise operation 2)
        # This is a fixed scaling operation that's deterministic
        out = out * self.in_channels
        
        return out


# Shape constants
IN_CHANNELS = 128
OUT_CHANNELS = 256
KERNEL_SIZE = 3
BATCH_SIZE = 8
HEIGHT = 512
WIDTH = 512
LEAKY_ALPHA = 0.01


def get_inputs():
    """Return input tensors for the forward pass"""
    # Create input tensor with deterministic values (zeros)
    x = torch.zeros(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    """Return arguments for __init__"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, True, LEAKY_ALPHA]