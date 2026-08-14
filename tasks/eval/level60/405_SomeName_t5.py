import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""
    
    def __init__(self, input_channels=128, output_channels=256):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Linear transformation to project input to proper dimension
        self.linear = nn.Linear(input_channels, output_channels, bias=False)
        
        # BatchNorm1d for normalization (will be set to eval mode)
        self.bn = nn.BatchNorm1d(output_channels)
        
        # Set BatchNorm to eval mode for deterministic behavior
        self.bn.eval()
    
    def forward(self, x):
        # Get batch size and sequence length from input
        batch_size, seq_len, _ = x.shape
        
        # Reshape to (batch_size * seq_len, input_channels) for linear layer
        x_reshaped = x.view(batch_size * seq_len, self.input_channels)
        
        # Apply linear transformation
        x_linear = self.linear(x_reshaped)
        
        # Reshape back to (batch_size, seq_len, output_channels)
        x_reshaped = x_linear.view(batch_size, seq_len, self.output_channels)
        
        # Transpose for BatchNorm1d: (batch_size, output_channels, seq_len)
        x_transposed = x_reshaped.transpose(1, 2)
        
        # Apply BatchNorm1d
        x_bn = self.bn(x_transposed)
        
        # Transpose back: (batch_size, seq_len, output_channels)
        x_bn = x_bn.transpose(1, 2)
        
        # Elementwise operations chain
        # 1. Add constant offset
        x_offset = x_bn + 0.1
        
        # 2. Multiply by scale factor
        x_scaled = x_offset * 1.5
        
        # 3. Apply ReLU activation
        x_relu = torch.relu(x_scaled)
        
        # 4. Apply softplus activation
        x_softplus = torch.nn.functional.softplus(x_relu)
        
        # 5. Apply tanh activation
        x_tanh = torch.tanh(x_softplus)
        
        # 6. Final scaling
        x_final = x_tanh * 2.0
        
        return x_final

# Module-level constants for shapes
INPUT_CHANNELS = 128
OUTPUT_CHANNELS = 256
BATCH_SIZE = 6
SEQ_LEN = 64

def get_inputs():
    """Return input tensors for the model."""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_CHANNELS)]

def get_init_inputs():
    """Return arguments for model initialization."""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS]