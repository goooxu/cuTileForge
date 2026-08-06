import torch
import torch.nn as nn

"""
TransformerBlockNorm (tier 3, norm)
"""


class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    def __init__(self, in_features, num_heads):
        super(Model, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        
        # LayerNorm for the transformer block
        self.ln1 = nn.LayerNorm(in_features)
        self.ln2 = nn.LayerNorm(in_features)
        
        # Simulated attention (not a real attention, just for structure)
        self.qkv_proj = nn.Linear(in_features, 3 * in_features, bias=False)
        self.out_proj = nn.Linear(in_features, in_features, bias=False)
        
        # MLP
        self.fc1 = nn.Linear(in_features, 4 * in_features)
        self.fc2 = nn.Linear(4 * in_features, in_features)
        self.dropout = nn.Dropout(0.1)
        
        # Ensure deterministic behavior
        self.ln1.eval()
        self.ln2.eval()

    def forward(self, x):
        # Save original for residual connections
        residual = x
        
        # LayerNorm + attention simulation
        x = self.ln1(x)
        
        # Simple linear projection to simulate attention (QKV)
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Simple attention computation (not softmax, just for structure)
        attn_output = torch.einsum('bnd,bmd->bnm', q, k)
        attn_output = attn_output / (self.in_features ** 0.5)
        
        # Apply attention to values
        x = torch.einsum('bnm,bmd->bnd', attn_output, v)
        x = self.out_proj(x)
        
        # Residual connection
        x = x + residual
        
        # MLP part
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Final residual connection
        x = x + residual
        
        return x


# Module-level constants for shape configuration
IN_FEATURES = 1024
NUM_HEADS = 16
BATCH_SIZE = 8
SEQ_LEN = 512

def get_inputs():
    # Return a list with a single tensor for the transformer block input
    return [torch.randn(BATCH_SIZE, SEQ_LEN, IN_FEATURES)]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_FEATURES, NUM_HEADS]