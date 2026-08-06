import torch
import torch.nn as nn

"""GatedConvReLU (tier 3, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 16
OUT_CHANNELS = 32
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
PADDING = 1

class Model(nn.Module):
    """GatedConvReLU (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Gate convolution (same shape as conv output)
        self.conv_gate = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # BatchNorm for convolution output (not gate)
        self.bn = nn.BatchNorm2d(out_channels)
        # Set to eval mode for deterministic behavior
        self.bn.eval()
        
        # Initialize weights with fixed values for reproducibility
        # Using kaiming normal initialization (default for Conv2d)
        # We'll override the default initialization to use small constant values
        with torch.no_grad():
            # Use a small value to ensure numerical stability
            torch.nn.init.constant_(self.conv.weight, 0.01)
            torch.nn.init.constant_(self.conv.bias, 0.01)
            torch.nn.init.constant_(self.conv_gate.weight, 0.01)
            torch.nn.init.constant_(self.conv_gate.bias, 0.01)
            torch.nn.init.constant_(self.bn.weight, 1.0)
            torch.nn.init.constant_(self.bn.bias, 0.0)
    
    def forward(self, x):
        # First convolution
        conv_out = self.conv(x)
        
        # BatchNorm applied to convolution output
        conv_out = self.bn(conv_out)
        
        # Gate computation (second convolution with same input)
        gate_out = self.conv_gate(x)
        
        # Sigmoid activation for gating
        gate_out = torch.sigmoid(gate_out)
        
        # Elementwise multiplication (gated convolution)
        out = conv_out * gate_out
        
        # Elementwise ReLU
        out = torch.relu(out)
        
        return out

def get_inputs():
    """Returns a list of input tensors for the model."""
    # Create input tensor with consistent values
    x = torch.ones(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
    # Add small random noise to avoid all-zero gradients during testing
    with torch.no_grad():
        x += 0.001 * torch.randn_like(x)
    return [x]

def get_init_inputs():
    """Returns arguments for Model.__init__."""
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        PADDING
    ]