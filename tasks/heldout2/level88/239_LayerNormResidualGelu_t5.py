import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormResidualGelu (tier 5, elementwise)"""
    
    def __init__(self, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LayerNorm modules for each layer
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size, elementwise_affine=True) 
            for _ in range(num_layers)
        ])
        
        # Create bias terms for residual connections
        self.residual_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size)) 
            for _ in range(num_layers)
        ])
        
        # Set to eval mode for deterministic behavior
        for ln in self.layernorms:
            ln.eval()

    def forward(self, x, residual_input):
        """
        Apply layer normalization, residual connection, and GELU activation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            residual_input: Residual input tensor of same shape
            
        Returns:
            Output tensor after normalization, residual add, and activation
        """
        outputs = []
        current = x
        
        for i in range(self.num_layers):
            # Apply layer normalization
            normalized = self.layernorms[i](current)
            
            # Add residual connection with bias
            residual = normalized + residual_input + self.residual_biases[i]
            
            # Apply GELU activation
            activated = torch.nn.functional.gelu(residual)
            
            outputs.append(activated)
            current = activated
            
        return outputs[-1] if outputs else current


# Module-level constants for shapes
BATCH_SIZE = 8
SEQ_LEN = 1024
HIDDEN_SIZE = 4096
NUM_LAYERS = 16

def get_inputs():
    """Returns input tensors for the model forward pass."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    residual_input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    return [x, residual_input]

def get_init_inputs():
    """Returns arguments for model initialization."""
    return [HIDDEN_SIZE, NUM_LAYERS]