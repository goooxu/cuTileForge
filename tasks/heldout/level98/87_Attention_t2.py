import torch
import torch.nn as nn

class Model(nn.Module):
    """Attention (tier 2, matmul)"""
    
    def __init__(self, batch_size, seq_len, num_heads, head_dim):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
    def forward(self, query, key, value):
        # Reshape inputs for attention computation
        # query: [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores: Q @ K^T
        # [B, num_heads, seq_len, head_dim] @ [B, num_heads, head_dim, seq_len]
        scores = torch.matmul(query, key.transpose(-1, -2))
        
        # Scale
        scores = scores * self.scale
        
        # Softmax over the last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Final output: attention_weights @ V
        # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_dim]
        output = torch.matmul(attention_weights, value)
        
        # Reshape back: [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)
        
        return output


# Module-level constants for shape configuration
BATCH_SIZE = 32
SEQ_LEN = 512
NUM_HEADS = 16
HEAD_DIM = 64

def get_inputs():
    """Generate deterministic input tensors for attention computation."""
    # Generate random but fixed tensors
    query = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    return [query, key, value]

def get_init_inputs():
    """Return configuration arguments for Model initialization."""
    return [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]