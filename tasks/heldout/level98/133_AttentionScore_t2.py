import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 2
SEQ_LEN = 4
HEADS = 2
DIM_PER_HEAD = 8

class Model(nn.Module):
    """AttentionScore (tier 2, conv)"""

    def __init__(self, heads, dim_per_head, seq_len):
        super(Model, self).__init__()
        self.heads = heads
        self.dim_per_head = dim_per_head
        self.seq_len = seq_len
        self.scale = 1.0 / (dim_per_head ** 0.5)

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(dim_per_head, dim_per_head, bias=False)
        self.k_proj = nn.Linear(dim_per_head, dim_per_head, bias=False)
        self.v_proj = nn.Linear(dim_per_head, dim_per_head, bias=False)

        # Use BatchNorm1d in eval mode to be deterministic
        self.bn = nn.BatchNorm1d(heads * dim_per_head)
        self.bn.eval()

    def forward(self, x):
        # x shape: (batch_size, seq_len, heads, dim_per_head)
        batch_size, seq_len, heads, dim_per_head = x.shape

        # Compute Q, K, V using individual projections per head
        # Reshape to (batch * seq_len, heads * dim) for batchnorm compatibility
        x_reshaped = x.view(batch_size * seq_len, heads * dim_per_head)
        x_norm = self.bn(x_reshaped).view(batch_size, seq_len, heads, dim_per_head)

        # Project to get Q, K, V
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Compute attention scores: Q @ K^T
        # q: (B, L, H, D), k: (B, L, H, D)
        # Reshape to (B*H, L, D) for matmul
        q_reshaped = q.permute(0, 2, 1, 3).reshape(batch_size * heads, seq_len, dim_per_head)
        k_reshaped = k.permute(0, 2, 1, 3).reshape(batch_size * heads, seq_len, dim_per_head)
        v_reshaped = v.permute(0, 2, 1, 3).reshape(batch_size * heads, seq_len, dim_per_head)

        # Scale and compute attention scores
        scores = torch.bmm(q_reshaped, k_reshaped.transpose(-1, -2)) * self.scale

        # Apply softmax along the last dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        result = torch.bmm(attn_weights, v_reshaped)

        # Reshape back to (B, L, H, D)
        result = result.reshape(batch_size, heads, seq_len, dim_per_head).permute(0, 2, 1, 3)

        # Return final result: (batch_size, seq_len, heads, dim_per_head)
        return result


def get_inputs():
    # Create input tensor with shape (BATCH_SIZE, SEQ_LEN, HEADS, DIM_PER_HEAD)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HEADS, DIM_PER_HEAD)
    return [x]


def get_init_inputs():
    return [HEADS, DIM_PER_HEAD, SEQ_LEN]