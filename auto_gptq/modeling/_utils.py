import json
import os
from logging import getLogger
from typing import List, Optional, Union

import accelerate
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig
from transformers.utils.hub import cached_file

from ..utils.import_utils import dynamically_import_QuantLinear
from ..utils.modeling_utils import recurse_setattr
from ._const import CPU, CUDA_0, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH, SUPPORTED_MODELS


logger = getLogger(__name__)


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
    use_qigen: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
    trainable: bool = False,
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
        disable_marlin=not use_marlin,
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
        use_qigen=use_qigen,
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
            if (not (desc_act) or group_size == -1) and not use_triton and not use_qigen:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
                    use_cuda_fp16=use_cuda_fp16,
                    trainable=trainable,
                    weight_dtype=submodule.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
                    trainable=trainable,
                    weight_dtype=submodule.weight.dtype,
                )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))

def preprocess_checkpoint_qigen(
    module,
    names,
    bits,
    group_size,
    checkpoint,
    name="",
):
    try:
        import cQIGen as qinfer
    except ImportError:
        logger.error("cQIGen not installed.")
        raise

    QuantLinear = dynamically_import_QuantLinear(
        use_triton=False,
        desc_act=False,
        group_size=group_size,
        bits=bits,
        disable_exllama=False,
        use_qigen=True,
    )
    if isinstance(module, QuantLinear):
        in_features = module.infeatures
        out_features = module.outfeatures

        zeros = checkpoint[name + ".qzeros"]
        scales = checkpoint[name + ".scales"].float()

        if zeros.dtype != torch.float32:
            new_zeros = torch.zeros_like(scales).float().contiguous()
            if bits == 4:
                qinfer.unpack_zeros4(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
            elif bits == 2:
                qinfer.unpack_zeros2(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
            elif bits == 3:
                logger.info("Unpacking zeros for 3 bits")
            new_scales = scales.contiguous()
        else:
            if scales.shape[1] != out_features:
                new_scales = scales.transpose(0, 1).contiguous()
            else:
                new_scales = scales.contiguous()
            if zeros.shape[1] != out_features:
                new_zeros = zeros.transpose(0, 1).contiguous()
            else:
                new_zeros = zeros.contiguous()

        checkpoint[name + ".zeros"], checkpoint[name + ".scales"] = (
            new_zeros,
            new_scales,
        )
        del checkpoint[name + ".qzeros"]
        del checkpoint[name + ".g_idx"]
        if name + ".bias" in checkpoint:
            checkpoint[name + ".bias"] = checkpoint[name + ".bias"].float()
        else:
            checkpoint[name + ".bias"] = torch.zeros(out_features)
        checkpoint_qweight = checkpoint[name + ".qweight"].int().contiguous()
        if bits == 4:
            qweight = torch.zeros(int(in_features // 8 * out_features)).int().contiguous()
            qinfer.pack4(
                checkpoint_qweight,
                qweight,
                in_features // 8,
                out_features,
                module.mb,
                module.tb,
                module.cutoff,
            )  # * (module.tt//tb))
        elif bits == 3:
            qweight = torch.zeros(int(in_features // 32 * 3 * out_features)).int().contiguous()
            qinfer.pack3(
                checkpoint_qweight,
                qweight,
                in_features // 32 * 3,
                out_features,
                module.mb // 32 * 3,
                module.tb,
                module.cutoff,
            )
        elif bits == 2:
            qweight = torch.zeros(int(in_features // 16 * out_features)).int().contiguous()
            qinfer.pack2(
                checkpoint_qweight,
                qweight,
                in_features // 16,
                out_features,
                module.mb,
                module.tb,
                module.cutoff,
            )  # * (module.tt//tb))
        checkpoint[name + ".qweight"] = qweight
        return

    for name1, child in module.named_children():
        preprocess_checkpoint_qigen(
            child,
            names,
            bits,
            group_size,
            checkpoint,
            name + "." + name1 if name != "" else name1,
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
    force_layer_back_to_cpu: bool = False,
    is_marlin_format: bool = False,
):
    QuantLinear = dynamically_import_QuantLinear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_exllama=False,
        disable_exllamav2=True,
        disable_marlin=not is_marlin_format,
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
        use_marlin=is_marlin_format,
    )
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
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

    ## exllamav2
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


def make_sure_no_tensor_in_meta_device(
    model, use_triton: bool, desc_act: bool, group_size: int, bits: int, disable_exllama: bool, disable_exllamav2: bool, use_marlin: bool = False,
):
    QuantLinear = dynamically_import_QuantLinear(use_triton, desc_act, group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2, disable_marlin=not use_marlin)
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear) and m.bias.device == torch.device("meta"):
            m.register_buffer("bias", torch.zeros((m.outfeatures), dtype=torch.float16, device="cpu"))


def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    assert bits == 4

    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(
        0,
        int_tensor.shape[1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    ).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)

    reverse_order_tensor = torch.arange(order_tensor.shape[0]).cuda()[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor


def unpack_awq(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        fp16_weight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros.cuda()
    qweight = awq_qweight.cuda()
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    infeatures = awq_qweight.shape[0]

    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    # zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)).to(
        torch.int16 if bits == 8 else torch.int8
    )
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)

    # Dequantize weights.
    scales = awq_scales.cuda()
    zeros = zeros.contiguous()
    scale_zeros = zeros * scales

    g_idx = torch.tensor([i // group_size for i in range(infeatures)], dtype=torch.int32)
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx].half()

    qdq_weight_T = weight * scale_mat - scale_zeros_mat.half()

    fp16_weight = qdq_weight_T.T.cuda()

    return fp16_weight, zeros


def pack_from_tensors(
    unpacked_qweight: torch.Tensor,
    unpacked_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        unpacked_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features)
        unpacked_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        qweight (`torch.LongTensor`):
            With shape (in_features // (32 // bits), out_features)
        qzeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features // (32 // bits))
    """
    assert bits == 4
    W = unpacked_qweight.clone().cpu()

    # TODO: This should be checked somehow.
    # if isinstance(linear, nn.Conv2d):
    #     W = W.flatten(1)
    # if isinstance(linear, transformers.pytorch_utils.Conv1D):
    #     W = W.t()

    awq_scales = awq_scales.t().contiguous()
    unpacked_qzeros = unpacked_qzeros.contiguous()
    unpacked_qzeros = unpacked_qzeros.cpu()

    awq_scales = awq_scales.cpu()
    scale_zeros = unpacked_qzeros.t() * awq_scales
    scales = awq_scales.clone()

    infeatures = unpacked_qweight.shape[1]

    intweight = []
    for idx in range(infeatures):
        g_idx = idx // group_size

        intweight.append(torch.round((W[:, idx] + scale_zeros[:, g_idx]) / scales[:, g_idx]).to(torch.int)[:, None])
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(np.uint32)

    i = 0
    row = 0
    qweight = np.zeros((intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        for j in range(i, i + (32 // bits)):
            qweight[row] |= intweight[j] << (bits * (j - i))
        i += 32 // bits
        row += 1

    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)

    unpacked_qzeros = unpacked_qzeros - 1
    torch.bitwise_and(unpacked_qzeros, (2**bits) - 1, out=unpacked_qzeros)

    unpacked_qzeros = unpacked_qzeros.numpy().astype(np.uint32)
    qzeros = np.zeros(
        (unpacked_qzeros.shape[0], unpacked_qzeros.shape[1] // 32 * bits),
        dtype=np.uint32,
    )
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + (32 // bits)):
            qzeros[:, col] |= unpacked_qzeros[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1

    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)

    return qweight, qzeros

def get_checkpoints(model_name_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs):
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

__all__ = [
    "get_device",
    "move_to_device",
    "find_layers",
    "get_module_by_name_prefix",
    "get_module_by_name_suffix",
    "make_quant",
    "preprocess_checkpoint_qigen",
    "pack_model",
    "autogptq_post_init",
    "check_and_get_model_type",
    "simple_dispatch_model",
    "make_sure_no_tensor_in_meta_device",
]
