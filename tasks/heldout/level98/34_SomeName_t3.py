import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

class Model(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(input_dim, input_dim, bias=False)
        
        # BatchNorm layer that needs to be set to eval mode
        self.norm = nn.BatchNorm1d(input_dim)
        self.norm.eval()

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()
        
        # Project inputs
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Reshape to (batch, seq_len, input_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.input_dim)
        
        # Project output
        output = self.o_proj(attn_output)
        
        # Apply batch normalization (eval mode ensures deterministic)
        # Reshape for batch norm: (batch * seq_len, input_dim)
        batch_size, seq_len, input_dim = output.size()
        output_flat = output.view(-1, input_dim)
        output_flat = self.norm(output_flat)
        output = output_flat.view(batch_size, seq_len, input_dim)
        
        return output


# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 4
INPUT_DIM = 8
NUM_HEADS = 2

def get_inputs():
    """Returns list of tensors to pass to forward: (query, key, value)"""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    return [query, key, value]

def get_init_inputs():
    """Returns list of arguments to pass to __init__"""
    return [INPUT_DIM, NUM_HEADS]