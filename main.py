import torch
import transformers


class MetadataTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, centered=False):
        self = torch.Tensor._make_subclass(cls, data)
        self.centered = centered
        return self

    def __getattr__(self, name):
        if name == 'centered':
            return False
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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


def apply_func_to_nested_tuple(t, func):
    """
    递归地对嵌套的 tuple 的每个元素应用函数 func，并保持原有的嵌套结构。

    :param t: 一个嵌套的 tuple
    :param func: 要应用到每个非 tuple 元素的函数
    :return: 返回一个结构与原 tuple 相同，但元素经过 func 操作的 tuple
    """
    if isinstance(t, tuple):
        # 如果是 tuple，递归地对每个元素应用函数
        return tuple(apply_func_to_nested_tuple(item, func) for item in t)
    else:
        # 如果不是 tuple，应用函数 func
        return func(t)


def get_shape(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.size())
    else:
        return None


def get_centered(t):
    if isinstance(t, MetadataTensor) and hasattr(t, 'centered'):
        return t.centered
    else:
        return None


indent = 0


def hook_pre_fn(module, inputs):
    global indent
    print('  ' * indent, '< ', module.__class__.__name__, '>')
    indent += 1

    inputs_centered = True

    def input_func(input):
        nonlocal inputs_centered
        if isinstance(input, MetadataTensor):
            inputs_centered = inputs_centered and input.centered
        else:
            inputs_centered = False
        print('  ' * indent, '<-', input.__class__.__name__, get_centered(input), get_shape(input))

    apply_func_to_nested_tuple(inputs, input_func)

    module._input_centered = inputs_centered

    if isinstance(module, torch.nn.LayerNorm):
        global ln_cnt
        ln_cnt += 1
        if inputs_centered:
            global foldable_cnt
            foldable_cnt += 1


def hook_fn(module, inputs, outputs):
    if isinstance(outputs, transformers.utils.ModelOutput):
        return outputs
    global indent

    single = False
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
        single = True

    def output_func(output):
        if isinstance(output, MetadataTensor):
            if isinstance(module, (
                torch.nn.LayerNorm,
                torch.nn.Linear,
            )) or module.__class__.__name__.startswith('Conv'):
                output.centered = True
            elif isinstance(module, torch.nn.Dropout):
                output.centered = module._input_centered
            print('  ' * indent, '->',  output.__class__.__name__, get_centered(output), get_shape(output))

        return output

    new_outputs = apply_func_to_nested_tuple(outputs, output_func)

    indent -= 1
    print('  ' * indent, '</', module.__class__.__name__, '>')

    if single:
        new_outputs = new_outputs[0]
    else:
        new_outputs = tuple(new_outputs)

    return new_outputs


def analyse(model_name: str):
    config = getattr(transformers, model_name + 'Config')()
    model = getattr(transformers, model_name + 'Model')(config)

    ln_cnt = 0
    foldable_cnt = 0

    hooks = []

    for layer in model.named_modules():
        hooks.append(layer[1].register_forward_pre_hook(hook_pre_fn))
        hooks.append(layer[1].register_forward_hook(hook_fn))

    input_ids = torch.randint(0, 1000, (1, 128))
    my_input_ids = MetadataTensor(input_ids, centered=False)
    out = model(my_input_ids)

    for hook in hooks:
        hook.remove()

    print('LayerNorm:', ln_cnt)
    print('Foldable:', foldable_cnt)


if __name__ == '__main__':
    analyse('Qwen2Moe')
