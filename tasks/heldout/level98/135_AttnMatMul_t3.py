import torch
import torch.nn as nn

"""AttnMatMul (tier 3, matmul)"""

# Module-level shape constants
BATCH_SIZE = 8
SEQ_LEN = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_HEADS

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.scale = HEAD_DIM ** -0.5
        
        # Q, K, V projections
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.key_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.value_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
    
    def forward(self, x):
        """
        Forward pass: compute attention scores, apply softmax, then matmul with values.
        
        Args:
            x: Input tensor of shape [BATCH_SIZE, SEQ_LEN, HIDDEN_DIM]
        
        Returns:
            Attention output of shape [BATCH_SIZE, SEQ_LEN, HIDDEN_DIM]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        q = self.query_proj(x)  # [B, S, H]
        k = self.key_proj(x)    # [B, S, H]
        v = self.value_proj(x)  # [B, S, H]
        
        # Reshape for multi-head attention: [B, S, H] -> [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # [B, num_heads, S, head_dim] @ [B, num_heads, head_dim, S] -> [B, num_heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax along the last dimension (sequence dimension)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values: attn_weights @ V
        # [B, num_heads, S, S] @ [B, num_heads, S, head_dim] -> [B, num_heads, S, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape back: [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim] -> [B, S, num_heads * head_dim]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, HIDDEN_DIM)
        
        # Project to output
        output = self.output_proj(context)
        
        return output

def get_inputs():
    """Return list of input tensors."""
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32),
    ]

def get_init_inputs():
    """Return list of arguments for __init__."""
    return []