import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 3, elementwise)"""

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Feed-forward network
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = (embed_dim // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.out_proj(context)
        
        # First residual connection
        x = residual + self.dropout(x)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Second residual connection
        x = residual + self.dropout(x)
        
        return x


# Module-level constants for shape configuration
BATCH_SIZE = 49
SEQ_LEN = 129
EMBED_DIM = 512
NUM_HEADS = 8
HIDDEN_DIM = 2049
DROPOUT = 0.1

def get_inputs():
    """Returns input tensors for the model forward pass."""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)]

def get_init_inputs():
    """Returns arguments for model initialization."""
    return [EMBED_DIM, NUM_HEADS, HIDDEN_DIM, DROPOUT]