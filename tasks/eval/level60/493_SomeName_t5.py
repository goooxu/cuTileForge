import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Create learnable parameters for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Initialize weights deterministically
        nn.init.normal_(self.q_proj.weight, mean=0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Permute to (batch, num_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        
        return attn_output


# Module-level constants for shapes
BATCH_SIZE = 3
SEQ_LEN = 4
EMBED_DIM = 8
NUM_HEADS = 2

def get_inputs():
    # Create input tensor with shape (batch_size, seq_len, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    return [x]

def get_init_inputs():
    # Return parameters for __init__
    return [EMBED_DIM, NUM_HEADS]