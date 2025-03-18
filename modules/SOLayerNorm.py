import torch
import torch.nn as nn
from torch.nn import functional as F

import rms_norm_cpp


def soln_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    output = rms_norm_cpp.forward(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )
    return output[0]


def replace_layer_norm_forward(
    layer: nn.LayerNorm,
    forward_fn: callable = soln_forward,
    class_name: str = 'SOLayerNorm'
) -> None:
    layer.__class__ = type(
        class_name,
        (nn.Module,),
        {'forward': forward_fn}
    )
    return None


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    my_schedule = schedule(
        wait=1000,
        warmup=500,
        active=2500,
    )

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

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=my_schedule,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            with record_function("LayerNorm"):
                for _ in range(12000):
                    model(x)
                    prof.step()
                print(model(x))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    replace_layer_norm_forward(model.norm, forward_fn=soln_forward)

    x -= x.mean()

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=my_schedule,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            with record_function("SOLayerNorm"):
                for _ in range(12000):
                    model(x)
                    prof.step()
                print(model(x))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
