import torch
import torch.nn as nn

"""NormResidualConv (tier 2, conv)"""

# Module-level constants for tensor shapes
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 32
BATCH_SIZE = 2
SEQ_LEN = 8
NUM_CHANNELS = 32

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self):
        super().__init__()
        
        # LayerNorm: normalise across features for each position
        self.norm = nn.LayerNorm(INPUT_DIM)
        
        # Conv1D: 1D convolution with kernel size 3 for local feature interaction
        self.conv1 = nn.Conv1d(
            in_channels=INPUT_DIM, 
            out_channels=HIDDEN_DIM, 
            kernel_size=3, 
            padding=1
        )
        
        # Additional conv to bring back to output dimension
        self.conv2 = nn.Conv1d(
            in_channels=HIDDEN_DIM, 
            out_channels=OUTPUT_DIM, 
            kernel_size=1
        )
        
        # BatchNorm for regularization - set to eval mode for determinism
        self.batch_norm = nn.BatchNorm1d(OUTPUT_DIM)
        self.batch_norm.eval()
        
        # Activation function
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        batch_size, seq_len, input_dim = x.shape
        
        # Ensure input has correct shape for operations
        assert input_dim == INPUT_DIM
        
        # Residual path: keep original for skip connection
        residual = x
        
        # Normalize across features (layer norm)
        x = self.norm(x)
        
        # Reshape for conv1d: (BATCH_SIZE, INPUT_DIM, SEQ_LEN)
        x = x.permute(0, 2, 1)
        
        # First convolution
        x = self.conv1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        
        # Apply batch norm (already in eval mode)
        x = self.batch_norm(x)
        
        # Reshape back to (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
        x = x.permute(0, 2, 1)
        
        # Add residual connection
        x = x + residual
        
        # Return the final output
        return x

def get_inputs():
    """Generate input tensor for the model"""
    # Create input tensor with shape (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    return [x]

def get_init_inputs():
    """Return initialization parameters (none needed for this model)"""
    return []