import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, query_dim=64, key_dim=64, value_dim=64, num_heads=1):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        head_dim = self.key_dim // self.num_heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        key = key.transpose(1, 2)      # (batch, heads, seq_len, head_dim)
        value = value.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, heads, seq_len, seq_len)
        scores = scores / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))
        
        # Apply softmax
        attn_weights = self.softmax(scores)
        
        # Compute output
        output = torch.matmul(attn_weights, value)  # (batch, heads, seq_len, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.value_dim)
        
        return output


# Module-level constants for shapes
QUERY_SHAPE = (2, 4, 64)
KEY_SHAPE = (2, 4, 64)
VALUE_SHAPE = (2, 4, 64)
NUM_HEADS = 1
KEY_DIM = 64
VALUE_DIM = 64

def get_inputs():
    """Returns a list of tensors to pass to forward"""
    query = torch.randn(QUERY_SHAPE)
    key = torch.randn(KEY_SHAPE)
    value = torch.randn(VALUE_SHAPE)
    return [query, key, value]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [QUERY_SHAPE[2], KEY_DIM, VALUE_DIM, NUM_HEADS]