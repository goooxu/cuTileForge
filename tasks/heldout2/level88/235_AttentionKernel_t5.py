import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 8
SEQ_LEN = 512
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM


class Model(nn.Module):
    """AttentionKernel (tier 5, conv)"""

    def __init__(self):
        super().__init__()
        # Project inputs to query, key, value
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        # Output projection
        self.out_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        # Scale factor for attention
        self.scale = HEAD_DIM ** -0.5
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Compute output: attn_probs @ V
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back to (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


def get_inputs():
    """Return input tensors for the forward pass."""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]


def get_init_inputs():
    """Return arguments for __init__."""
    return []