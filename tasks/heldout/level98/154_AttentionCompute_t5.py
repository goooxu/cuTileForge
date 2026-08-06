import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionCompute (tier 5, matmul)"""

    def __init__(self, head_dim: int = 64, num_heads: int = 1, seq_len: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v are (B, seq_len, head_dim * num_heads)
        B, seq_len, _ = q.shape
        
        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(B, self.num_heads, seq_len, self.head_dim)
        k = k.view(B, self.num_heads, seq_len, self.head_dim)
        v = v.view(B, self.num_heads, seq_len, self.head_dim)
        
        # Compute attention scores: (B, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attention = torch.softmax(scores, dim=-1)
        
        # Compute output: (B, num_heads, seq_len, head_dim)
        out = torch.matmul(attention, v)
        
        # Reshape back to (B, seq_len, num_heads * head_dim)
        out = out.view(B, seq_len, self.num_heads * self.head_dim)
        
        return out


BATCH_SIZE = 2
SEQ_LEN = 64
HEAD_DIM = 64
NUM_HEADS = 1

def get_inputs():
    q = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM * NUM_HEADS)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM * NUM_HEADS)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM * NUM_HEADS)
    return [q, k, v]

def get_init_inputs():
    return [HEAD_DIM, NUM_HEADS, SEQ_LEN]