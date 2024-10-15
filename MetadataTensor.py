import torch


class MetadataTensor(torch.Tensor):
    def __new__(cls, data, centered=False):
        # instance = torch.Tensor._make_subclass(cls, data)
        instance = torch.Tensor.as_subclass(torch.as_tensor(data), cls)
        instance.centered = centered
        return instance

    # def __init__(self, data, centered=False):
    #     self.centered = centered

    def __add__(self, other):
        result = super(MetadataTensor, self).__add__(other)
        if isinstance(other, MetadataTensor):
            centered = self.centered and other.centered
        else:
            centered = False
        return MetadataTensor(result, centered)

    def __repr__(self):
        if hasattr(self, "centered"):
            centered = self.centered
        else:
            centered = None
        return f"MetadataTensor({super().__repr__()}, centered={centered})"


x = MetadataTensor(torch.tensor([1, 2, 3]), centered=True)
print(x.size())
y = torch.tensor([4, 5, 6])
z = x + y
print(z)
