import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 5
SEQ_LEN = 5
IN_CHANNELS = 8
OUT_CHANNELS = 8
KERNEL_SIZE = 3

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv1d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE//2)
        
        # Batch normalization with eval mode for determinism
        self.bn = nn.BatchNorm1d(OUT_CHANNELS)
        self.bn.eval()
        
        # Layer normalization for normalization step
        self.norm = nn.LayerNorm([OUT_CHANNELS, SEQ_LEN])
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x shape: (BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
        
        # Apply convolution
        out = self.conv(x)
        
        # Apply batch normalization
        out = self.bn(out)
        
        # Apply layer normalization
        out = self.norm(out)
        
        # Apply activation
        out = self.activation(out)
        
        # Residual connection: add original input
        out = out + x
        
        return out

def get_inputs():
    """Return list of tensors to pass to forward"""
    # Create input tensor with shape (BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)]

def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return []