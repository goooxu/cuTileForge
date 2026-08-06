import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, matmul)"""

    def __init__(self, batch_size, seq_len, hidden_dim, num_heads):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Initialize learnable parameters for Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Pre-compute scale factor
        self.scale = 1.0 / torch.sqrt(torch.tensor(hidden_dim // num_heads, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project to Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, hidden_dim)
        K = self.W_k(x)  # (batch_size, seq_len, hidden_dim)
        V = self.W_v(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.hidden_dim // self.num_heads)
        K = K.view(batch_size, seq_len, self.num_heads, self.hidden_dim // self.num_heads)
        V = V.view(batch_size, seq_len, self.num_heads, self.hidden_dim // self.num_heads)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention: score = (Q @ K.T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        scores = scores * self.scale
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.W_o(attention_output)
        
        return output


# Module-level constants for shape dimensions
BATCH_SIZE = 4
SEQ_LEN = 512
HIDDEN_DIM = 1024
NUM_HEADS = 16

def get_inputs():
    """Return input tensors for forward pass"""
    # Create input tensor with shape (batch_size, seq_len, hidden_dim)
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32)
    ]

def get_init_inputs():
    """Return arguments for __init__"""
    return [
        BATCH_SIZE,
        SEQ_LEN,
        HIDDEN_DIM,
        NUM_HEADS
    ]