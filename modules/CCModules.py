import torch.nn as nn
from .CCConv1D import center_conv1d
from .CCEmbedding import center_embedding
from .CCLinear import center_linear
from transformers.pytorch_utils import Conv1D


def center_modules(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        center_linear(layer)
    elif isinstance(layer, Conv1D):
        center_conv1d(layer)
    elif isinstance(layer, nn.Embedding):
        center_embedding(layer)
    return None
