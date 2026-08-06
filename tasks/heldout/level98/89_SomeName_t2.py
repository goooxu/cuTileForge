import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

Q_DIM = 256
KV_DIM = 256
SEQ_LEN = 1024
BATCH_SIZE = 8

class Model(nn.Module):
    """SomeName (tier 2, matmul)"""
    
    def __init__(self, q_dim, kv_dim, seq_len, batch_size):
        super(Model, self).__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Projection matrices (no training needed, just for matrix sizes)
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, kv_dim, bias=False)
        
        # Use dropout but set to eval mode for deterministic behavior
        self.dropout = nn.Dropout(p=0.1)
        self.dropout.eval()
        
    def forward(self, query, key, value):
        # Scaled dot-product attention computation
        # (batch_size, seq_len, q_dim) @ (batch_size, seq_len, q_dim).T 
        # -> (batch_size, seq_len, seq_len)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.q_dim, dtype=torch.float32))
        
        # Softmax over the last dimension (sequence length)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Output: attn_weights @ V -> (batch_size, seq_len, kv_dim)
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    query = torch.randn(BATCH_SIZE, SEQ_LEN, Q_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, KV_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, KV_DIM)
    return [query, key, value]

def get_init_inputs():
    return [Q_DIM, KV_DIM, SEQ_LEN, BATCH_SIZE]