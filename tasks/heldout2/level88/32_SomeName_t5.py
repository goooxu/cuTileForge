import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_DIM = 256
HIDDEN_DIM = 512
NUM_TOKENS = 64

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, input_dim, hidden_dim, num_tokens):
        super(Model, self).__init__()
        
        # Convolutional layer for feature transformation
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # For deterministic behavior
        self.norm1.eval()
        self.norm2.eval()
    
    def forward(self, x):
        # x shape: (batch_size, num_tokens, input_dim)
        batch_size = x.shape[0]
        
        # Permute to (batch_size, input_dim, num_tokens) for conv1d
        x_conv = x.permute(0, 2, 1)
        
        # First convolution
        out = self.conv1(x_conv)
        out = self.activation(out)
        
        # Second convolution
        out = self.conv2(out)
        
        # Permute back to (batch_size, num_tokens, input_dim)
        out = out.permute(0, 2, 1)
        
        # Add residual connection
        out = out + x
        
        # Apply normalization with residual
        out = self.norm1(out)
        
        # Additional processing with normalization
        out = self.norm2(out)
        
        return out

def get_inputs():
    # Create input tensor with shape (batch_size, num_tokens, input_dim)
    batch_size = 4
    return [torch.randn(batch_size, NUM_TOKENS, INPUT_DIM)]

def get_init_inputs():
    # Return arguments for model initialization
    return [INPUT_DIM, HIDDEN_DIM, NUM_TOKENS]