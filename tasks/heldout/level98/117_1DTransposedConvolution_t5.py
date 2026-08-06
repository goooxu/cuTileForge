import torch
import torch.nn as nn

"""1D Transposed Convolution (tier 5, conv)"""

class Model(nn.Module):
    """1D Transposed Convolution (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, padding, dilation):
        super(Model, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, 
                                        padding, output_padding, dilation, bias=True)
        self.conv1.eval()

    def forward(self, x):
        return self.conv1(x)


# Module-level constants for shape configuration
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 5
STRIDE = 3
OUTPUT_PADDING = 2
PADDING = 1
DILATION = 2
BATCH_SIZE = 16
SEQ_LEN = 128

def get_inputs():
    """Returns list of input tensors for the forward pass."""
    # Shape: (batch_size, in_channels, seq_len)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [x]

def get_init_inputs():
    """Returns list of arguments for Model constructor."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, OUTPUT_PADDING, PADDING, DILATION]