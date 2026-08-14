import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 4
SEQ_LENGTH = 64
NUM_HEADS = 8
HEAD_DIM = 64

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, num_heads, head_dim):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, q, k, v):
        # Compute attention scores
        scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Reshape for multi-head attention
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        
        # Compute attention scores: Q @ K^T
        # q: (batch_size, seq_len, num_heads, head_dim)
        # k: (batch_size, seq_len, num_heads, head_dim)
        # Transpose k to (batch_size, num_heads, head_dim, seq_len) for matmul
        
        # Permute to (batch_size, num_heads, seq_len, head_dim) for easier manipulation
        q_permuted = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k_permuted = k.permute(0, 2, 3, 1)  # (batch_size, num_heads, head_dim, seq_len)
        
        # Compute scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q_permuted, k_permuted)
        scores = scores * scale
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Reshape v for matmul: (batch_size, num_heads, seq_len, head_dim)
        v_permuted = v.permute(0, 2, 1, 3)
        
        # Compute output: attention_weights @ v
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        # v_permuted: (batch_size, num_heads, seq_len, head_dim)
        # result: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, v_permuted)
        
        # Permute back to (batch_size, seq_len, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3)
        
        return output


def get_inputs():
    # Generate input tensors with deterministic values
    q = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, HEAD_DIM)
    k = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, HEAD_DIM)
    v = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, HEAD_DIM)
    return [q, k, v]


def get_init_inputs():
    return [NUM_HEADS, HEAD_DIM]