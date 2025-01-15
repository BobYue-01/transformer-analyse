import torch
import torch.nn as nn
from torch.nn import functional as F


def native_soln_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    return F.rms_norm(
        x,
        self.normalized_shape,
        self.weight,
        self.eps
    ) + self.bias


def myln_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
    normalized_tensor = (x - mean) / (torch.sqrt(var) + self.eps)
    if self.elementwise_affine:
        normalized_tensor = normalized_tensor * self.weight + self.bias
    return normalized_tensor


def soln_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    norm_x = x.norm(2, dim=-1, keepdim=True) * x.size(-1) ** -.5
    normalized_tensor = x / (norm_x + self.eps)
    if self.elementwise_affine:
        normalized_tensor = normalized_tensor * self.weight + self.bias
    return normalized_tensor


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
        wait=100,
        warmup=50,
        active=250,
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

    model = MyModel().cuda()
    x = torch.randn(10).cuda()

    replace_layer_norm_forward(model.norm, forward_fn=myln_forward)

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=my_schedule,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            with record_function("LayerNorm"):
                for _ in range(1200):
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
                for _ in range(1200):
                    model(x)
                    prof.step()
                print(model(x))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
