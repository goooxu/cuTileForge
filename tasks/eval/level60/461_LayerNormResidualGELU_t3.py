import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormResidualGELU (tier 3, norm)"""

    def __init__(self, hidden_size=1024, num_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LayerNorm modules for each layer
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Set all LayerNorms to eval mode for deterministic behavior
        for ln in self.layernorms:
            ln.eval()
        
        # Linear projection for residual connection (shared across layers)
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
        
        # Set to eval mode
        self.residual_proj.eval()

    def forward(self, x):
        # Process through all layers
        for i in range(self.num_layers):
            # Layer normalization
            normalized = self.layernorms[i](x)
            
            # Residual connection: add original input
            residual = x
            
            # Apply GELU activation
            activated = nn.functional.gelu(normalized + residual)
            
            # Update x for next iteration
            x = activated
        
        return x


# Module-level constants for tensor shapes
BATCH_SIZE = 6
SEQ_LENGTH = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 12

def get_inputs():
    """Generate input tensors for the model"""
    # Create input tensor with shape (batch_size, seq_length, hidden_size)
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [HIDDEN_SIZE, NUM_LAYERS]