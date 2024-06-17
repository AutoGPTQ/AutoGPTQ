import json
import logging
import os
from logging import getLogger
from typing import List, Optional, Union

import accelerate
import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig
from transformers.utils.hub import cached_file

from ..quantization import BaseQuantizeConfig
from ..utils.import_utils import dynamically_import_QuantLinear
from ..utils.modeling_utils import recurse_setattr
from ._const import CPU, CUDA_0, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH, SUPPORTED_MODELS

logger = getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Optional[Union[torch.Tensor, nn.Module]], device: torch.device):
    if obj is None:
        return obj
    else:
        if get_device(obj) != device:
            obj = obj.to(device)
        return obj


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
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
    name="",
    use_triton: bool = False,
    use_marlin: bool = False,
    disable_exllama: Optional[bool] = None,
    disable_exllamav2: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
):
    # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
    if disable_exllama is None:
        if disable_exllamav2:
            disable_exllama = False
        else:
            disable_exllama = True

    QuantLinear = dynamically_import_QuantLinear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        use_marlin=use_marlin,
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
    )

    if isinstance(module, QuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            if (not (desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    use_cuda_fp16=use_cuda_fp16,
                    weight_dtype=submodule.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=submodule.weight.dtype,
                )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))


def convert_gptq_v1_to_v2_format(
    model,
    quantize_config: BaseQuantizeConfig,
    qlinear_kernel: nn.Module,
):
    # Limit thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        for _, submodule in model.named_modules():
            # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
            # additions here do not overflow.
            # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
            # overflow ~<=13% based on testing
            if isinstance(submodule, qlinear_kernel):
                if quantize_config.bits == 2:
                    submodule.qzeros.data += 0b01010101010101010101010101010101
                elif quantize_config.bits == 3:
                    submodule.qzeros.data[:, range(0, submodule.qzeros.data.shape[1], 3)] += (
                        0b00100100100100100100100100100100
                    )
                    submodule.qzeros.data[:, range(1, submodule.qzeros.data.shape[1], 3)] += (
                        0b10010010010010010010010010010010
                    )
                    submodule.qzeros.data[:, range(2, submodule.qzeros.data.shape[1], 3)] += (
                        0b01001001001001001001001001001001
                    )
                elif quantize_config.bits == 4:
                    submodule.qzeros.data += 0b00010001000100010001000100010001
                elif quantize_config.bits == 8:
                    submodule.qzeros.data += 0b00000001000000010000000100000001
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    return model


def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: BaseQuantizeConfig,
    qlinear_kernel: nn.Module,
):
    # Limit thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        for _, submodule in model.named_modules():
            # sym=False has underflow probability of ~<=13% during testing. No underflow possible for sym=True.
            if isinstance(submodule, qlinear_kernel):
                if quantize_config.bits == 2:
                    submodule.qzeros.data -= 0b01010101010101010101010101010101
                elif quantize_config.bits == 3:
                    submodule.qzeros.data[:, range(0, submodule.qzeros.data.shape[1], 3)] -= (
                        0b00100100100100100100100100100100
                    )
                    submodule.qzeros.data[:, range(1, submodule.qzeros.data.shape[1], 3)] -= (
                        0b10010010010010010010010010010010
                    )
                    submodule.qzeros.data[:, range(2, submodule.qzeros.data.shape[1], 3)] -= (
                        0b01001001001001001001001001001001
                    )
                elif quantize_config.bits == 4:
                    submodule.qzeros.data -= 0b00010001000100010001000100010001
                elif quantize_config.bits == 8:
                    submodule.qzeros.data -= 0b00000001000000010000000100000001
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    return model


def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    warmup_triton: bool = False,
    force_layer_back_to_cpu: bool = False,
    use_marlin: bool = False,
):
    QuantLinear = dynamically_import_QuantLinear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_exllama=False,
        disable_exllamav2=True,
        use_marlin=use_marlin,
    )

    if force_layer_back_to_cpu:
        model.to(CPU)

    logger.info("Packing model...")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
        use_triton=use_triton,
        use_cuda_fp16=use_cuda_fp16,
        desc_act=desc_act,
        disable_exllama=False,
        disable_exllamav2=True,
        use_marlin=use_marlin,
    )
    qlayers = find_layers(model, [QuantLinear])

    # Limit pack() thread usage to avoid auto-parallizataion regression
    with tctl.threadpool_limits(limits=1):
        pbar = tqdm(qlayers.keys(), leave=True)
        for name in pbar:
            pbar.set_description(f"Packing {name}")

            quantizers[name], scale, zero, g_idx = quantizers[name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to(CPU)
            layers[name], scale, zero, g_idx = (
                layers[name].to(CPU),
                scale.to(CPU),
                zero.to(CPU),
                g_idx.to(CPU),
            )
            if QuantLinear.QUANT_TYPE == "marlin":
                qlayers[name].pack(layers[name], scale)
            else:
                qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)

    logger.info("Model packed.")

    if use_triton and warmup_triton:
        logger.warning(
            "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
        )
        QuantLinear.warmup(model.to(CUDA_0), seqlen=model.seqlen)
    return QuantLinear


def check_and_get_model_type(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
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


def autogptq_post_init(model, use_act_order: bool, max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    for name, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1,
                }

            if not use_act_order:
                submodule._use_act_order = False
            else:
                submodule._use_act_order = True

            # Disable this heuristic for detecting act_order, but it could be used instead of the config.
            """
            if submodule.g_idx is None:
                submodule.act_order = False
            elif submodule.g_idx is not None and ((submodule.g_idx == 0).all() or torch.equal(submodule.g_idx.cpu(), torch.tensor([i // submodule.group_size for i in range(submodule.g_idx.shape[0])], dtype=torch.int32))):
                submodule.g_idx = None
                submodule.act_order = False
            else:
                submodule.act_order = True
            """

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(
                device_to_buffers_size[device]["max_dq_buffer_size"],
                submodule.qweight.numel() * 8,
            )

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(
                    device_to_buffers_size[device]["max_inner_outer_dim"],
                    submodule.infeatures,
                    submodule.outfeatures,
                )

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        try:
            from exllama_kernels import prepare_buffers, set_tuning_params
        except ImportError as e:
            raise ImportError(
                f"Could not import exllama backend dependencies prepare_buffers, set_tuning_params with the following error: {e}"
            )

        device_to_buffers = {}

        if use_act_order:
            if max_input_length is None:
                max_input_len = EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_len = max_input_length
        else:
            if max_input_length is not None:
                logger.info(
                    "Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored."
                )
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros(
                    (max_input_len, buffers_size["max_inner_outer_dim"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "temp_dq": torch.zeros(
                    (1, buffers_size["max_dq_buffer_size"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }

        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers

        for device, buffers in model.device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        # The buffers need to have been initialized first before calling make_q4.
        for name, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
                submodule.post_init()

    # exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from ..nn_modules.qlinear.qlinear_exllamav2 import ExLlamaV2DeviceTensors

        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
                device = submodule.qweight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model


def get_checkpoints(
    model_name_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs
):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_name_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_name_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    # The model is sharded over several checkpoints.
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_name_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                shard_index = cached_file(
                    model_name_or_path,
                    shard_index_name,
                    **cached_file_kwargs,
                )
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    # The model is sharded over several checkpoints.
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        # Download the shards from the index.json.
                        shards = list(set(index_json["weight_map"].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(
                                model_name_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_name_or_path,
                        possible_model_basename + ext,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename

    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
        )

    return False, resolved_archive_file, true_model_basename


# return the most stable tensor dtype for quantization while minimizing vram
def auto_dtype_from_config(config: PretrainedConfig) -> torch.dtype:
    dtype = getattr(config, "torch_dtype")
    if not dtype or not isinstance(dtype, torch.dtype):
        raise ValueError("Your model config.json does not have torch_dtype set. Please check for model " "corruption.")

    if dtype == torch.float32:
        return torch.bfloat16
    elif dtype == torch.float16:
        return torch.float16
    else:
        # up/down-cast everything else to bfloat16 if not already in bfloat16
        return torch.bfloat16


__all__ = [
    "get_device",
    "auto_dtype_from_config",
    "move_to_device",
    "find_layers",
    "get_module_by_name_prefix",
    "get_module_by_name_suffix",
    "make_quant",
    "pack_model",
    "autogptq_post_init",
    "check_and_get_model_type",
    "simple_dispatch_model",
    "convert_gptq_v1_to_v2_format",
    "convert_gptq_v2_to_v1_format",
]
