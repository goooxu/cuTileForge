import torch
import torch.nn as nn

"""QKVAttention (tier 3, matmul)"""
class Model(nn.Module):
    """QKVAttention (tier 3, matmul)"""
    
    def __init__(self, batch_size, num_heads, seq_len, head_dim):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        
        # Initialize linear projections for Q, K, V
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # Initialize scaling factor
        self.scale = torch.tensor(head_dim ** -0.5)
        
    def forward(self, x):
        batch_size, seq_len, head_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, head_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, head_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, head_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim // self.num_heads)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim // self.num_heads)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim // self.num_heads)
        
        # Transpose for attention computation: (batch_size, num_heads, seq_len, head_dim//num_heads)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale the scores
        scores = scores * self.scale
        
        # Apply softmax along the last dimension
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values: attn_weights @ V
        context = torch.matmul(attn_weights, v)
        
        # Reshape and return: (batch_size, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, head_dim)
        
        return context

# Module-level constants for shape configuration
BATCH_SIZE = 4
NUM_HEADS = 16
SEQ_LEN = 512
HEAD_DIM = 128

def get_inputs():
    # Generate input tensor with proper shape
    # Shape: (batch_size, seq_len, head_dim)
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM)]

def get_init_inputs():
    return [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]