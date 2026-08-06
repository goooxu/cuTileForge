import torch
import torch.nn as nn

class Model(nn.Module):
    """QKVAttention (tier 3, matmul)"""
    def __init__(self, n_head=8, n_embd=64):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.scale = 1.0 / (n_embd ** 0.5)
    
    def forward(self, q, k, v):
        batch_size, seq_len, _ = q.shape
        k_t = k.transpose(-1, -2)
        scores = torch.matmul(q, k_t)
        scores = scores * self.scale
        scores = scores - torch.max(scores, dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores)
        attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
        output = torch.matmul(attention_weights, v)
        return output

BATCH_SIZE = 4
SEQ_LEN = 8
N_HEAD = 8
N_EMBD = 64

def get_inputs():
    q = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEAD * N_EMBD)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEAD * N_EMBD)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEAD * N_EMBD)
    return [q, k, v]

def get_init_inputs():
    return [N_HEAD, N_EMBD]