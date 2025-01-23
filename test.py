import torch
import time

# 大张量测试
x = torch.randn(1000000).cuda()

# 方法 1: torch.rsqrt
start = time.time()
for _ in range(100000):
    result = torch.rsqrt(torch.sum(x.pow(2)))
print(result)
print("torch.rsqrt:", time.time() - start)

# 方法 2: 1 / torch.norm
start = time.time()
for _ in range(100000):
    result = 1 / torch.norm(x, p=2)
print(result)
print("1 / torch.norm:", time.time() - start)
