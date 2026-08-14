import torch
import torch.nn as nn


class Model(nn.Module):
    """HuberLoss (tier 5, loss)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        return torch.nn.functional.huber_loss(predictions, targets)


batch_size = 4472
dim = 4472
def get_inputs():
    return [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
