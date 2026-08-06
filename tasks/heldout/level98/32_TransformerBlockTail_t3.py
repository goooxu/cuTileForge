import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlockTail (tier 3, conv)"""
    
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(intermediate_size)
        
        # Convolutional projection
        self.fc1 = nn.Conv1d(hidden_size, intermediate_size, 1)
        self.fc2 = nn.Conv1d(intermediate_size, hidden_size, 1)
        
        # Activation function
        self.act = nn.GELU()
        
        # Pooling layer
        self.pool = nn.AvgPool1d(2, stride=2)
        
        # Set batchnorm to eval mode for determinism
        self.norm1.eval()
        self.norm2.eval()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        
        # First normalization and residual
        x_norm1 = self.norm1(x)
        
        # Conv1d requires (batch, channels, seq_len)
        x_conv = x_norm1.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        
        # First projection
        x_proj = self.fc1(x_conv)  # (batch, intermediate_size, seq_len)
        x_proj = self.act(x_proj)
        
        # Pooling
        x_pooled = self.pool(x_proj)  # (batch, intermediate_size, seq_len // 2)
        
        # Second projection
        x_out = self.fc2(x_pooled)  # (batch, hidden_size, seq_len // 2)
        
        # Residual connection with proper shape handling
        # Need to pool x_norm1 to match the reduced sequence length
        x_residual = self.pool(x_norm1.permute(0, 2, 1))  # (batch, hidden_size, seq_len // 2)
        
        # Add residual and normalize
        x_output = x_out + x_residual
        x_output = x_output.permute(0, 2, 1)  # Back to (batch, seq_len // 2, hidden_size)
        
        return x_output


# Module-level constants for shape configuration
BATCH_SIZE = 32
SEQ_LEN = 64
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_HEADS = 8


def get_inputs():
    # Return input tensors for forward pass
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]


def get_init_inputs():
    # Return arguments for __init__
    return [HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS]