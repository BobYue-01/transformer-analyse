import torch
import torch.nn as nn
from torch.nn import functional as F


def soln_forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, self.normalized_shape, self.weight, self.eps) + self.bias


def replace_layer_norm(layer: torch.nn.LayerNorm) -> None:
    if isinstance(layer, torch.nn.LayerNorm):
        layer.__class__ = type('SOLayerNorm', (nn.Module,), {'forward': soln_forward})
    return None


if __name__ == '__main__':
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super(MyModel, self).__init__()
            self.embed_dim = 10
            self.eps = 1e-5
            self.norm = torch.nn.LayerNorm(self.embed_dim, eps=self.eps)
            self.norm.weight.data = torch.randn(self.embed_dim)
            self.norm.bias.data = torch.randn(self.embed_dim)

        def forward(self, x):
            return self.norm(x)

    model = MyModel()
    x = torch.randn(10)

    print(model(x))

    replace_layer_norm(model.norm)

    x = x - x.mean()
    print(model(x))
