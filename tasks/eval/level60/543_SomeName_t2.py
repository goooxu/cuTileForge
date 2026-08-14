import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""
    
    def __init__(self, input_channels=64, output_channels=128):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Learnable parameters for the elementwise operations
        self.scale1 = nn.Parameter(torch.ones(input_channels))
        self.scale2 = nn.Parameter(torch.ones(input_channels))
        self.scale3 = nn.Parameter(torch.ones(output_channels))
        self.bias1 = nn.Parameter(torch.zeros(input_channels))
        self.bias2 = nn.Parameter(torch.zeros(input_channels))
        self.bias3 = nn.Parameter(torch.zeros(output_channels))
        
    def forward(self, x, y):
        # Chain of elementwise operations
        # 1. Elementwise multiply
        out1 = x * y
        
        # 2. Elementwise add
        out2 = out1 + self.bias1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # 3. Elementwise multiply with scale
        out3 = out2 * self.scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # 4. Elementwise relu
        out4 = torch.relu(out3)
        
        # 5. Elementwise square
        out5 = out4 ** 2
        
        return out5

# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
BATCH_SIZE = 6
HEIGHT = 96
WIDTH = 96
def get_inputs():
    """Return list of tensors for forward pass"""
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    y = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    return [x, y]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]