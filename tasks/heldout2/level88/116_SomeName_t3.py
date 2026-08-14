import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shapes
INPUT_DIM = 256
HIDDEN_DIM = 512
NUM_HEADS = 8
DROPOUT_P = 0.1

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super(Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Convolution layers with proper normalization
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(input_dim)
        
        # Evaluation mode for batch normalization
        self.norm1.eval()
        self.norm2.eval()
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_len)
        original_x = x
        
        # First convolution block: conv -> norm -> activation
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Second convolution block: conv -> norm
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        x = x + self.residual_weight * original_x
        
        # Final activation
        x = self.activation(x)
        
        return x

def get_inputs():
    # Create sample input tensor with shape (batch_size, input_dim, seq_len)
    batch_size = 4
    input_dim = INPUT_DIM
    seq_len = 64
    
    # Create a deterministic input tensor
    x = torch.randn(batch_size, input_dim, seq_len)
    return [x]

def get_init_inputs():
    # Return arguments for model initialization
    return [INPUT_DIM, HIDDEN_DIM, NUM_HEADS]