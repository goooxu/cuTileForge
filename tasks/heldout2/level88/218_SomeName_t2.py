import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_FEATURES = 16
OUTPUT_FEATURES = 16
BATCH_SIZE = 2
SEQ_LEN = 8

class Model(nn.Module):
    """SomeName (tier 2, norm)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # LayerNorm for normalization
        self.layernorm = nn.LayerNorm(OUTPUT_FEATURES)
        # BatchNorm1d for residual path (will be set to eval mode)
        self.batchnorm = nn.BatchNorm1d(OUTPUT_FEATURES)
        # Activation function
        self.activation = nn.ReLU()
        
        # Set BatchNorm to eval mode for deterministic behavior
        self.batchnorm.eval()
        
    def forward(self, input_tensor, residual_tensor):
        # Apply LayerNorm
        normalized = self.layernorm(input_tensor)
        
        # Add residual connection
        added = normalized + residual_tensor
        
        # Apply activation function
        output = self.activation(added)
        
        return output

def get_inputs():
    """Generate input tensors for the model."""
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, OUTPUT_FEATURES)
    residual_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, OUTPUT_FEATURES)
    return [input_tensor, residual_tensor]

def get_init_inputs():
    """Return empty list since __init__ takes no arguments."""
    return []