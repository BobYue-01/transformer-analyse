import torch
from transformers.utils import ModelOutput
from .Counter import Counter
from .MetadataTensor import MetadataTensor


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


def get_last_modules(t):
    if isinstance(t, MetadataTensor) and hasattr(t, 'last_modules'):
        return t.last_modules
    else:
        return set()


def create_hook_fns(counter: Counter):
    def hook_pre_fn(module, inputs):
        print('  ' * counter.indent, '< ', module.__class__.__name__, '>')
        counter.indent += 1
        for name, sub_module in module.named_children():
            print('  ' * counter.indent, name, ':', sub_module.__class__.__name__)

        inputs_centered = True
        last_modules = set()

        def input_func(input):
            nonlocal inputs_centered
            nonlocal last_modules
            if isinstance(input, MetadataTensor):
                inputs_centered = inputs_centered and input.centered
                last_modules |= input.last_modules
            else:
                inputs_centered = False
            print('  ' * counter.indent, '<-', input.__class__.__name__, get_centered(input), get_shape(input), len(get_last_modules(input)), get_last_modules(input))

        apply_func_to_nested_tuple(inputs, input_func)

        module._input_centered = inputs_centered
        module._last_modules = last_modules

        if 'LayerNorm' in module.__class__.__name__:
            counter.ln_cnt += 1
            if inputs_centered:
                counter.layernorms.add(module)
                counter.foldable_cnt += 1
                counter.center_modules |= last_modules

    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, ModelOutput):
            return outputs

        single = False
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
            single = True

        def output_func(output):
            if not isinstance(output, MetadataTensor) and isinstance(output, torch.Tensor):
                output = MetadataTensor(output, centered=False)
            if isinstance(output, MetadataTensor):
                module_name = module.__class__.__name__
                if 'LayerNorm' in module_name or 'Linear' in module_name or 'Conv' in module_name or 'Embedding' in module_name:
                    output.centered = True
                    output.last_modules = {module}
                elif 'Dropout' in module_name:
                    output.centered = module._input_centered
                    output.last_modules = module._last_modules
                print('  ' * counter.indent, '->',  output.__class__.__name__, get_centered(output), get_shape(output), len(get_last_modules(output)), get_last_modules(output))

            return output

        new_outputs = apply_func_to_nested_tuple(outputs, output_func)

        counter.indent -= 1
        print('  ' * counter.indent, '</', module.__class__.__name__, '>')

        if single:
            new_outputs = new_outputs[0]
        else:
            new_outputs = tuple(new_outputs)

        return new_outputs

    return hook_pre_fn, hook_fn