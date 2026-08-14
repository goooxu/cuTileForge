import torch
import torch.nn as nn

# Module-level constants for shapes
INPUT_FEATURES = 16
HIDDEN_FEATURES = 32
OUTPUT_FEATURES = 16
BATCH_SIZE = 2
SEQ_LEN = 3

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        
        # Convolutional layers for the transformer-style block
        self.conv1 = nn.Conv1d(input_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(hidden_features, output_features, 1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(input_features)
        self.norm2 = nn.LayerNorm(output_features)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Set evaluation mode for deterministic behavior
        self.norm1.eval()
        self.norm2.eval()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        
        # Residual connection
        residual = x
        
        # Normalize
        x = self.norm1(x)
        
        # Transpose for conv1d: (batch_size, input_features, seq_len)
        x = x.permute(0, 2, 1)
        
        # First convolution
        x = self.conv1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        
        # Transpose back: (batch_size, output_features, seq_len) -> (batch_size, seq_len, output_features)
        x = x.permute(0, 2, 1)
        
        # Add residual (pad or truncate if necessary)
        if x.shape[-1] != residual.shape[-1]:
            # Pad or truncate residual to match output features
            if residual.shape[-1] < x.shape[-1]:
                pad_width = x.shape[-1] - residual.shape[-1]
                residual = torch.cat([residual, torch.zeros_like(residual[:, :, :pad_width])], dim=-1)
            else:
                residual = residual[:, :, :x.shape[-1]]
        
        x = x + residual
        
        # Final normalization
        x = self.norm2(x)
        
        return x

def get_inputs():
    # Return list of input tensors
    return [torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_FEATURES)]

def get_init_inputs():
    # Return list of arguments for __init__
    return [INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES]