import torch
import transformers
import random
import time
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from contextlib import redirect_stdout
import utils
import modules
import argparse
from torch.profiler import profile, record_function, ProfilerActivity, schedule

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default='GPT2', help='Model name')
parser.add_argument("--skip_equality", action="store_true", help='Check equality of output')
parser.add_argument("--abs_tol", type=float, default=1e-5, help='Absolute tolerance')
parser.add_argument("--replace", action="store_true", help='Replace output')
parser.add_argument("--skip_speed", action="store_true", help='Speed comparison')
parser.add_argument("--speed_iter", type=int, default=16, help='Speed iteration')

args = parser.parse_args()

Config: transformers.PretrainedConfig = getattr(transformers, args.model_name + 'Config')
Model: transformers.PreTrainedModel = getattr(transformers, args.model_name + 'Model')

config = Config()
my_ln_model = Model(config).cuda()
my_so_ln_model = Model(config).cuda()

my_so_ln_model.load_state_dict(my_ln_model.state_dict())
my_ln_model.eval()
my_so_ln_model.eval()

input_ids = torch.randint(0, 1000, (1, 128)).cuda()
my_input_ids = utils.MetadataTensor(input_ids, centered=False).cuda()

folded_counter = utils.Counter()
original_counter = utils.Counter()

folded_hook_pre_fn, folded_hook_fn = utils.create_analyse_hook_fns(folded_counter, _print=True)
original_hook_pre_fn, original_hook_fn = utils.create_analyse_hook_fns(original_counter, _print=True)

with open(f"./log/{args.model_name}_Count.txt", "w") as f:
    with redirect_stdout(f):

        with utils.HookManager(my_so_ln_model, folded_hook_fn, folded_hook_pre_fn):
            my_so_ln_model(my_input_ids)

        with utils.HookManager(my_ln_model, original_hook_fn, original_hook_pre_fn):
            my_ln_model(my_input_ids)

        for layer in folded_counter.center_modules:
            modules.center_modules(layer)

        for layer in folded_counter.layernorms:
            modules.replace_layer_norm_forward(
                layer,
                forward_fn=modules.soln_forward,
                class_name='MySOLayerNorm'
            )

        for layer in original_counter.layernorms:
            modules.replace_layer_norm_forward(
                layer,
                forward_fn=modules.myln_forward,
                class_name='MyLayerNorm'
            )

with open(f"./log/{args.model_name}_Count.txt", "w") as f:
    with redirect_stdout(f):
        print('LayerNorm:', folded_counter.ln_cnt)
        print('Foldable:', folded_counter.foldable_cnt)
        print('Center modules:', folded_counter.center_modules)

for layer in folded_counter.layernorms:
    modules.replace_layer_norm_forward(layer)

for layer in folded_counter.center_modules:
    modules.center_modules(layer)

output_queue = []
check = utils.Check(all_abs_tol=args.abs_tol, tolerate_bias=True)


def check_close_and_replace(tensor_a, tensor_b, check: utils.Check, tensor_a_str, tensor_b_str):
    check.hide_val()
    locals()[tensor_a_str] = tensor_a
    locals()[tensor_b_str] = tensor_b
    equal, bias = check.check_eq(tensor_a_str, tensor_b_str, abs_tol=args.abs_tol, local_vars=locals())
    if (
        equal
        and args.replace
        and isinstance(tensor_a, torch.Tensor)
        and isinstance(tensor_b, torch.Tensor)
        and bias is None
    ):
            tensor_b.data = tensor_a.data
    check.show_val()


def apply_func_to_nested_tuple_pair(t1, t2, func, *args, **kwargs):
    if isinstance(t1, tuple) and isinstance(t2, tuple):
        return tuple(apply_func_to_nested_tuple_pair(x1, x2, func, *args, **kwargs) for x1, x2 in zip(t1, t2))
    else:
        return func(t1, t2, *args, **kwargs)


def hook_original(module, input, output):
    name = module.__class__.__name__
    output_queue.append((output, name))

    # if isinstance(output, tuple):
    #     output = output[0]

    # with torch._tensor_str.printoptions(precision=10, sci_mode=True):
    #     len_shape = len(output.shape)
    #     index = tuple([0] * (len_shape - 2) + [slice(None, 4), slice(None, 4)])
    #     print(module.__class__.__name__, output[index])


def hook_folded(module, input, output):
    folded_name = module.__class__.__name__ + '_folded'
    original_output, original_name = output_queue.pop(0)
    original_name += '_original'
    apply_func_to_nested_tuple_pair(original_output, output, check_close_and_replace, check, original_name, folded_name)

    # if isinstance(output, tuple):
    #     output0 = output[0]

    # with torch._tensor_str.printoptions(precision=10, sci_mode=True):
    #     len_shape = len(output0.shape)
    #     index = tuple([0] * (len_shape - 2) + [slice(None, 4), slice(None, 4)])
    #     print(module.__class__.__name__, output0[index])


if not args.skip_equality:
    with open(f"./log/{args.model_name}_Compare.txt", "w") as f:
        with redirect_stdout(f):
            with utils.HookManager(my_ln_model, hook_original, None, list(my_ln_model.modules())[1:]):
                original_out = my_ln_model(input_ids)
            with utils.HookManager(my_so_ln_model, hook_folded, None, list(my_so_ln_model.modules())[1:]):
                folded_out = my_so_ln_model(input_ids)
            check.check_eq('folded_out[0]', 'original_out[0]', local_vars=locals())
            check.summary()


if not args.skip_speed:
    import json

    index = [
        "count",
        "node_id",
        "is_async",
        "is_remote",
        "use_device",
        "cpu_time_total",
        "device_time_total",
        "self_cpu_time_total",
        "self_device_time_total",
        "input_shapes",
        "stack",
        "scope",
        "cpu_memory_usage",
        "device_memory_usage",
        "self_cpu_memory_usage",
        "self_device_memory_usage",
        "cpu_children",
        "cpu_parent",
        "device_type",
        "is_legacy",
        "flops",
    ]

    for _ in range(args.speed_iter):

        json_result = {}
        current_name = ""

        def export_json(prof):
            stats = prof.key_averages()
            json_result[current_name] = {}
            for stat in stats:
                json_result[current_name][stat.key] = {}
                for i in index:
                    data = str(eval(f"stat.{i}"))
                    json_result[current_name][stat.key][i] = data


        my_schedule = schedule(
            wait=100,
            warmup=50,
            active=250,
        )

        start_time = time.strftime("%Y%m%d-%H%M%S")

        torch.cuda.empty_cache()
        with torch.no_grad():
            current_name = 'folded'
            with profile(
                activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
                ],
                profile_memory=True,
                record_shapes=True,
                schedule=my_schedule,
                with_flops=True,
                on_trace_ready=export_json
            ) as prof:
                for _ in range(400):
                    my_so_ln_model(input_ids)
                    prof.step()

            current_name = 'original'
            with profile(
                activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
                ],
                profile_memory=True,
                record_shapes=True,
                schedule=my_schedule,
                with_flops=True,
                on_trace_ready=export_json
            ) as prof:
                for _ in range(400):
                    my_ln_model(input_ids)
                    prof.step()

        with open(f"./log/{args.model_name}_Speed_{start_time}.json", "w") as f:
            json.dump(json_result, f, indent=4)
