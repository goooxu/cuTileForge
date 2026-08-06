import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 3, elementwise)"""

    def __init__(self, seq_len, hidden_dim, num_heads, batch_size, seq_dim, hidden_dim_inner):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_dim = seq_dim
        self.hidden_dim_inner = hidden_dim_inner
        
        # LayerNorm for normalization (deterministic in eval mode)
        self.norm = nn.LayerNorm(hidden_dim)
        self.norm.eval()
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
    def forward(self, x, residual):
        """
        Perform LayerNorm, residual connection, activation, and output projection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            residual: Residual tensor of same shape as x
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # LayerNorm
        normalized = self.norm(x)
        
        # Residual connection
        residual_added = normalized + residual
        
        # GELU activation
        activated = nn.functional.gelu(residual_added)
        
        # Output projection (linear layer)
        output = self.out_proj(activated)
        
        return output

# Module-level constants
SEQ_LEN = 512
HIDDEN_DIM = 768
NUM_HEADS = 12
BATCH_SIZE = 8
SEQ_DIM = 1
HIDDEN_DIM_INNER = 768

def get_inputs():
    """Return input tensors for the forward pass"""
    # Create tensors with the specified dimensions
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    residual = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x, residual]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [SEQ_LEN, HIDDEN_DIM, NUM_HEADS, BATCH_SIZE, SEQ_DIM, HIDDEN_DIM_INNER]