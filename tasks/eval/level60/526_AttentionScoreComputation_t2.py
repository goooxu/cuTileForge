import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScoreComputation (tier 2, matmul)"""
    
    def __init__(self, query_dim, key_dim, value_dim, num_heads, seq_len):
        super(Model, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        
        # Compute head dimensions
        self.head_dim = key_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, key_dim)
        self.k_proj = nn.Linear(query_dim, key_dim)
        self.v_proj = nn.Linear(query_dim, value_dim)
        
        # Output projection
        self.out_proj = nn.Linear(value_dim, value_dim)
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, query, key, value):
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        batch_size = query.shape[0]
        q = q.view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale
        scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        q = q * scale
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, self.seq_len, self.value_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 6
SEQ_LEN = 32
QUERY_DIM = 64
KEY_DIM = 64
VALUE_DIM = 64
NUM_HEADS = 4

def get_inputs():
    """Returns list of input tensors for the forward pass"""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, QUERY_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, VALUE_DIM)
    return [query, key, value]

def get_init_inputs():
    """Returns list of arguments for __init__"""
    return [QUERY_DIM, KEY_DIM, VALUE_DIM, NUM_HEADS, SEQ_LEN]