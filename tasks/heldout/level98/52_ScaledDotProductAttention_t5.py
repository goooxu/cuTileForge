import torch
import torch.nn as nn

# Shape configuration constants
BATCH_SIZE = 16
SEQ_LEN = 1024
HIDDEN_DIM = 64
NUM_HEADS = 32

class Model(nn.Module):
    """ScaledDotProductAttention (tier 5, matmul)"""

    def __init__(self, batch_size, seq_len, hidden_dim, num_heads):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Pre-project queries, keys, values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Scale factor: 1/sqrt(head_dim)
        self.scale = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, x):
        """Compute scaled dot-product attention scores.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project inputs
        q = self.q_proj(x)  # (batch, seq, hidden)
        k = self.k_proj(x)  # (batch, seq, hidden)
        v = self.v_proj(x)  # (batch, seq, hidden)
        
        # Reshape for multi-head attention: (batch, num_heads, seq, head_dim)
        head_dim = hidden_dim // self.num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention scores: (batch, num_heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Softmax over the last dimension (key sequence dimension)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values: (batch, num_heads, seq, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, seq, num_heads, head_dim) -> (batch, seq, hidden)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return out


def get_inputs():
    """Generate input tensors for the model."""
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32)
    ]


def get_init_inputs():
    """Return initialization arguments for the model."""
    return [BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, NUM_HEADS]