import torch
import torch.nn as nn

"""NormActResidual (tier 3, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layernorm modules
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Linear projections for residual connection
        self.linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        
        # Set to eval mode for deterministic behavior
        for ln in self.layernorms:
            ln.eval()

    def forward(self, x):
        # Input x shape: (B, S, H) = (batch, seq_len, hidden)
        batch_size, seq_len, hidden_size = x.shape
        
        # Initialize output with same shape as input
        output = torch.zeros(batch_size, seq_len, hidden_size, dtype=x.dtype, device=x.device)
        
        # Process each layer
        for i in range(self.num_layers):
            # Layernorm
            ln_out = self.layernorms[i](x)
            
            # Linear projection
            linear_out = self.linears[i](ln_out)
            
            # Residual connection
            x = x + linear_out
            
            # Activation (GELU-like operation without randomness)
            output = torch.nn.functional.gelu(x, approximate='tanh')
        
        return output


# Module-level constants
BATCH_SIZE = 8
SEQ_LEN = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 6

def get_inputs():
    # Generate deterministic input tensors
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32)
    return [x]

def get_init_inputs():
    # Configuration for __init__
    return [HIDDEN_SIZE, NUM_LAYERS]