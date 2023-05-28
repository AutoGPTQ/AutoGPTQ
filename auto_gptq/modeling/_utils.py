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
    for layer in layers:
        if isinstance(module,layer):
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


def make_quant(
    module,
    names,
    bits,
    group_size,
    name='',
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    trainable=False
):
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size)

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            ori_layer_device = get_device(getattr(module, attr))
            delattr(module, attr)
            if isinstance(tmp,nn.Linear):
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif isinstance(tmp,nn.Conv2d):
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif isinstance(tmp,transformers.pytorch_utils.Conv1D):            
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            if (not(desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinear(
                    bits, group_size, in_features, out_features, True, use_cuda_fp16=use_cuda_fp16, trainable=trainable
                )
            else:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, True, trainable=trainable)
            new_layer.device = ori_layer_device
            setattr(module, attr, new_layer.to(ori_layer_device))
    for name1, child in module.named_children():
        make_quant(
            child,
            names,
            bits,
            group_size,
            name + '.' + name1 if name != '' else name1,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=desc_act,
            trainable=trainable
        )


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
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def make_sure_no_tensor_in_meta_device(model, use_triton, desc_act, group_size):
    QuantLinear = dynamically_import_QuantLinear(use_triton, desc_act, group_size)
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear) and m.bias.device == torch.device("meta"):
            m.register_buffer('bias', torch.zeros((m.outfeatures), dtype=torch.float16, device="cpu"))


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
    "make_sure_no_tensor_in_meta_device"
]
