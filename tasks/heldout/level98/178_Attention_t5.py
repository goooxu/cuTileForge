import torch
import torch.nn as nn

class Model(nn.Module):
    """Attention (tier 5, matmul)"""
    
    def __init__(self, heads, seq_len, hidden_dim):
        super(Model, self).__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Initialize linear layers for Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights deterministically
        for layer in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.seq_len, self.heads, self.hidden_dim // self.heads)
        k = k.view(batch_size, self.seq_len, self.heads, self.hidden_dim // self.heads)
        v = v.view(batch_size, self.seq_len, self.heads, self.hidden_dim // self.heads)
        
        # Transpose to (batch, heads, seq, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: (batch, heads, seq, seq)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale by sqrt(d_head)
        d_head = self.hidden_dim // self.heads
        attn_scores = attn_scores / torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        
        # Softmax along the last dimension
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute output: (batch, heads, seq, d_head)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        output = attn_output.view(batch_size, self.seq_len, self.hidden_dim)
        
        return output

# Module-level constants for shapes
HEADS = 8
SEQ_LEN = 128
HIDDEN_DIM = 512
BATCH_SIZE = 32

def get_inputs():
    """Returns input tensors for forward pass."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Returns arguments for __init__."""
    return [HEADS, SEQ_LEN, HIDDEN_DIM]