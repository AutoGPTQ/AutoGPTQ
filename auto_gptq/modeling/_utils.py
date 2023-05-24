from logging import getLogger
from typing import Union

import accelerate
import torch
import torch.nn as nn
from transformers import AutoConfig
import transformers

from ._const import SUPPORTED_MODELS, CPU, CUDA_0
from ..utils.import_utils import dynamically_import_QuantLinear


logger = getLogger(__name__)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Union[torch.Tensor, nn.Module], device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


def find_layers(module, layers=None, name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_module_by_name_prefix(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def make_quant(module, names, bits, group_size, name='', use_triton=False, use_cuda_fp16=True, desc_act=False):
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size)

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            ori_layer_device = get_device(getattr(module, attr))
            delattr(module, attr)
            if type(tmp) == nn.Linear:
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif type(tmp) == nn.Conv2d:
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif type(tmp) == transformers.pytorch_utils.Conv1D:            
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            if (not(desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, True, use_cuda_fp16=use_cuda_fp16)
            else:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, True)
            new_layer.device = ori_layer_device
            setattr(module, attr, new_layer.to(ori_layer_device))
    for name1, child in module.named_children():
        make_quant(child, names, bits, group_size, name + '.' + name1 if name != '' else name1, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16,desc_act=desc_act)


def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    warmup_triton: bool = False,
    force_layer_back_to_cpu: bool = False
):
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size)

    if force_layer_back_to_cpu:
        model.to(CPU)

    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16, desc_act=desc_act)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx = layers[name].to(CPU), scale.to(CPU), zero.to(CPU), g_idx.to(CPU)
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)
    logger.info('Model packed.')

    if use_triton and warmup_triton:
        logger.warning(
            "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
        )
        QuantLinear.warmup(model.to(CUDA_0), seqlen=model.seqlen)


def check_and_get_model_type(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


def simple_dispatch_model(model, device_map):
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    prev_hook = None
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]
    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d == "cpu":
            _, prev_hook = accelerate.cpu_offload_with_hook(
                m,
                execution_device=main_device,
                prev_module_hook=prev_hook
            )
        else:
            d = torch.device(d)
            accelerate.hooks.attach_align_device_hook(m, execution_device=d)
            prev_hook = None
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def make_sure_not_tensor_in_meta_device(model, use_triton, desc_act, group_size):
    QuantLinear = dynamically_import_QuantLinear(use_triton, desc_act, group_size)
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear) and m.bias.device == torch.device("meta"):
            m.register_buffer('bias', torch.zeros((m.outfeatures), dtype=torch.float16, device="cpu"))


# this function is temporarily added, will be removed once the feature is added into accelerate
def add_align_logits_hook_to_model(model, device_map: dict):
    from accelerate.hooks import AlignDevicesHook, CpuOffload, add_hook_to_module
    from accelerate.utils import find_device, send_to_device, named_module_tensors, set_module_tensor_to_device
    from typing import Mapping

    skip_keys = ("past_key_values", "layer_past", "attention_mask", "position_ids")

    def send_to_device_except(data, device, non_blocking=False, skip_keys=()):
        if isinstance(data, Mapping):
            new_data = {
                k: (v if k in skip_keys else send_to_device(v, device, non_blocking))
                for k, v in data.items()
            }
            return type(data)(new_data)
        else:
            return send_to_device(data, device, non_blocking)

    class AlignLogitsHook(AlignDevicesHook):

        def pre_forward(self, module, *args, **kwargs):
            if self.io_same_device:
                self.input_device = find_device([args, kwargs])
            if self.offload:
                for name, _ in named_module_tensors(
                    module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                ):
                    set_module_tensor_to_device(module, name, self.execution_device, value=self.weights_map[name])

            return (
                send_to_device(args, self.execution_device),
                send_to_device_except(kwargs, self.execution_device, skip_keys=skip_keys),
            )

        def post_forward(self, module, output):
            if self.offload:
                for name, _ in named_module_tensors(
                    module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                ):
                    set_module_tensor_to_device(module, name, "meta")
            if self.io_same_device and self.input_device is not None:
                output = send_to_device_except(output, self.input_device, skip_keys=skip_keys)
            return output

    for n, d in device_map.items():
        if n == "":
            continue
        m = get_module_by_name_suffix(model, n)
        if hasattr(m, "_hf_hook"):
            old_hook = m._hf_hook
            if isinstance(old_hook, AlignDevicesHook):
                logger.debug(f"replace the original hook AlignDevicesHook with AlignLogitsHook at {n}")
                hook = AlignLogitsHook(
                    execution_device=old_hook.execution_device,
                    offload=old_hook.offload,
                    io_same_device=True,
                    weights_map=old_hook.weights_map,
                    offload_buffers=old_hook.offload,
                    place_submodules=old_hook.place_submodules
                )
                add_hook_to_module(m, hook, append=False)
            elif isinstance(old_hook, CpuOffload):
                logger.debug(f"append AlignLogitsHook to {n} (prev hook is CpuOffload)")
                hook = AlignLogitsHook(execution_device=old_hook.execution_device, io_same_device=True)
                add_hook_to_module(m, hook, append=True)


__all__ = [
    "get_device",
    "move_to_device",
    "find_layers",
    "get_module_by_name_prefix",
    "get_module_by_name_suffix",
    "make_quant",
    "pack_model",
    "check_and_get_model_type",
    "simple_dispatch_model",
    "make_sure_not_tensor_in_meta_device",
    "add_align_logits_hook_to_model"
]
