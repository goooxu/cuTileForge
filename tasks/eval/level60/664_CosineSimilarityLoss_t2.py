import torch
import torch.nn as nn


class Model(nn.Module):
    """CosineSimilarityLoss (tier 2, loss)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        return torch.mean(1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=-1))


batch_size = 48
dim = 384
def get_inputs():
    return [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
