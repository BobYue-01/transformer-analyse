import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from ..modules.SOLayerNorm import soln_forward, replace_layer_norm_forward


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

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():

            with record_function("LayerNorm"):
                model(x)

            replace_layer_norm_forward(model.norm)

            with record_function("SOLayerNorm"):
                model(x)