import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScoreComputation (tier 3, matmul)"""
    
    def __init__(self, batch_size, seq_len, num_heads, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, query, key, value):
        # Compute attention scores: scale, softmax, matmul
        # query: (batch_size, num_heads, seq_len, head_dim)
        # key: (batch_size, num_heads, head_dim, seq_len)
        
        # Scale factor for attention
        scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key) * scale
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute output: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 512
NUM_HEADS = 16
HEAD_DIM = 64

def get_inputs():
    """Generate input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN, dtype=torch.float32)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    return [query, key, value]

def get_init_inputs():
    """Return arguments for __init__ method"""
    return [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]