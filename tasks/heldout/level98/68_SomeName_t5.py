import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    def __init__(self, head_dim=64, seq_len=128, num_heads=8, hidden_size=512):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Project to Q, K, V using conv1d (matches tier 5, conv pattern)
        self.q_proj = nn.Conv1d(hidden_size, head_dim * num_heads, kernel_size=1)
        self.k_proj = nn.Conv1d(hidden_size, head_dim * num_heads, kernel_size=1)
        self.v_proj = nn.Conv1d(hidden_size, head_dim * num_heads, kernel_size=1)
        
        # Output projection
        self.o_proj = nn.Conv1d(head_dim * num_heads, hidden_size, kernel_size=1)
        
        # Scale factor for attention
        self.scale = 1.0 / (head_dim ** 0.5)
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to Q, K, V (B, hidden, L) -> (B, head_dim * num_heads, L)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: (B, num_heads, head_dim, L)
        Q = Q.view(batch_size, self.num_heads, self.head_dim, self.seq_len)
        K = K.view(batch_size, self.num_heads, self.head_dim, self.seq_len)
        V = V.view(batch_size, self.num_heads, self.head_dim, self.seq_len)
        
        # Compute attention scores: (B, num_heads, L, L)
        scores = torch.matmul(Q.transpose(-1, -2), K) * self.scale
        
        # Apply softmax along last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values: (B, num_heads, L, head_dim)
        output = torch.matmul(attention_weights, V.transpose(-1, -2))
        
        # Reshape back: (B, head_dim * num_heads, L)
        output = output.transpose(-1, -2).contiguous().view(batch_size, self.head_dim * self.num_heads, self.seq_len)
        
        # Project output: (B, hidden, L)
        return self.o_proj(output)

# Module-level constants for shapes
BATCH_SIZE = 4
HIDDEN_SIZE = 512
SEQ_LEN = 128
NUM_HEADS = 8
HEAD_DIM = 64

def get_inputs():
    # Create input tensor with shape (BATCH_SIZE, HIDDEN_SIZE, SEQ_LEN)
    return [torch.randn(BATCH_SIZE, HIDDEN_SIZE, SEQ_LEN)]

def get_init_inputs():
    # Initialize with configured parameters
    return []