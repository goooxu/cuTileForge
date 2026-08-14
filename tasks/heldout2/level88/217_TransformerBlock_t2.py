import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 2, elementwise)"""
    
    def __init__(self, embed_dim=2048, hidden_dim=8192, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Linear projections (simulating attention and MLP)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # MLP projections
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # GELU activation
        self.gelu = nn.GELU()
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        
        # First layer norm and residual path
        residual = x
        x = self.norm1(x)
        
        # Compute Q, K, V projections
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.embed_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Simulated attention (using simple sum for throughput testing)
        # In a real transformer, this would be proper attention computation
        x = q + k + v  # Elementwise operation for throughput
        
        # Projection
        x = self.proj(x)
        
        # First residual connection
        x = x + residual
        
        # Second layer norm and MLP path
        residual = x
        x = self.norm2(x)
        
        # MLP forward
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        
        # Second residual connection
        x = x + residual
        
        return x

# Module-level constants for shape configuration
BATCH_SIZE = 8
SEQ_LEN = 1024
EMBED_DIM = 2048
HIDDEN_DIM = 8192
NUM_HEADS = 16

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [EMBED_DIM, HIDDEN_DIM, NUM_HEADS]