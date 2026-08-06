import torch
import torch.nn as nn

class Model(nn.Module):
    """AttnCore (tier 5, matmul)"""

    def __init__(self, num_heads, seq_len, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, query, key, value):
        batch_size, seq_len, d_model = query.shape
        
        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output: A @ V
        output = torch.matmul(attention_weights, value)
        
        # Reshape back to (batch, seq_len, d_model)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output

# Constants for shape dimensions
BATCH_SIZE = 32
SEQ_LEN = 512
NUM_HEADS = 16
D_MODEL = 1024

def get_inputs():
    """Generate input tensors for the attention computation."""
    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL),
        torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL),
        torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    ]

def get_init_inputs():
    """Generate initialization arguments for the model."""
    return [NUM_HEADS, SEQ_LEN, D_MODEL]