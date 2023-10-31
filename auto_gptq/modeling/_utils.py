from logging import getLogger
from typing import Union, Optional

import accelerate
import torch
import torch.nn as nn
from transformers import AutoConfig
import transformers

from ._const import SUPPORTED_MODELS, CPU, CUDA_0, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
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
    use_triton: bool = False,
    disable_exllama: Optional[bool] = None,
    disable_exllamav2: bool = False, 
    use_qigen: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
    trainable: bool = False
):
    # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
    if disable_exllama is None:
        if disable_exllamav2:
            disable_exllama = False
        else:
            disable_exllama = True

    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2, use_qigen=use_qigen)

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
            if (not(desc_act) or group_size == -1) and not use_triton and not use_qigen:
                new_layer = QuantLinear(
                    bits, group_size, in_features, out_features, True, use_cuda_fp16=use_cuda_fp16, trainable=trainable, weight_dtype=tmp.weight.dtype
                )
            else:
                new_layer = QuantLinear(bits, group_size, in_features, out_features, True, trainable=trainable, weight_dtype=tmp.weight.dtype)
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
            trainable=trainable,
            disable_exllama=disable_exllama,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen
        )

def preprocess_checkpoint_qigen(
    module,
    names,
    bits,
    group_size,
    checkpoint,
    name='',
):
    try:
        import cQIGen as qinfer
    except ImportError:
        logger.error('cQIGen not installed.')
        raise

    QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=False, group_size=group_size, bits=bits, disable_exllama=False, use_qigen=True)
    if isinstance(module, QuantLinear):
        in_features = module.infeatures
        out_features = module.outfeatures
        
        zeros = checkpoint[name + '.qzeros']
        scales = checkpoint[name + '.scales'].float()
        
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
                new_scales = scales.transpose(0,1).contiguous()
            else:
                new_scales = scales.contiguous()
            if zeros.shape[1] != out_features:
                new_zeros = zeros.transpose(0,1).contiguous()
            else:
                new_zeros = zeros.contiguous()

        checkpoint[name + '.zeros'],checkpoint[name + '.scales'] = new_zeros, new_scales
        del checkpoint[name + '.qzeros']
        del checkpoint[name + '.g_idx']
        if name + '.bias' in checkpoint:
            checkpoint[name + '.bias'] = checkpoint[name + '.bias'].float()
        else:
            checkpoint[name + '.bias'] = torch.zeros(out_features)
        checkpoint_qweight = checkpoint[name + '.qweight'].int().contiguous()
        if bits == 4:
            qweight = torch.zeros(int(in_features // 8 * out_features)).int().contiguous()
            qinfer.pack4(checkpoint_qweight, qweight, in_features // 8, out_features, module.mb, module.tb, module.cutoff)# * (module.tt//tb))
        elif bits == 3:
            qweight = torch.zeros(int(in_features // 32 * 3 * out_features)).int().contiguous()
            qinfer.pack3(checkpoint_qweight, qweight, in_features // 32 * 3, out_features, module.mb // 32 * 3, module.tb, module.cutoff)
        elif bits == 2:
            qweight = torch.zeros(int(in_features // 16 * out_features)).int().contiguous()
            qinfer.pack2(checkpoint_qweight, qweight, in_features // 16, out_features, module.mb, module.tb, module.cutoff)# * (module.tt//tb))
        checkpoint[name + '.qweight'] = qweight
        return

    for name1, child in module.named_children():
        preprocess_checkpoint_qigen(
            child,
            names,
            bits,
            group_size,
            checkpoint,
            name + '.' + name1 if name != '' else name1,
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
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=False, disable_exllamav2=True)

    if force_layer_back_to_cpu:
        model.to(CPU)

    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton, use_cuda_fp16=use_cuda_fp16, desc_act=desc_act, disable_exllama=False, disable_exllamav2=True)
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
                    "max_inner_outer_dim": 1
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

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(device_to_buffers_size[device]["max_dq_buffer_size"], submodule.qweight.numel() * 8)

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(device_to_buffers_size[device]["max_inner_outer_dim"], submodule.infeatures, submodule.outfeatures)

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        from exllama_kernels import prepare_buffers, set_tuning_params
        
        device_to_buffers = {}

        if use_act_order:
            if max_input_length is None:
                max_input_len = EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_len = max_input_length
        else:
            if max_input_length is not None:
                logger.info("Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored.")
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros((max_input_len, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
                "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
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
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device,0))

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
                submodule.post_init(temp_dq = model.device_tensors[device])
    torch.cuda.empty_cache()

    return model


def make_sure_no_tensor_in_meta_device(model, use_triton, desc_act, group_size, bits: int):
    QuantLinear = dynamically_import_QuantLinear(use_triton, desc_act, group_size, bits=bits)
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
    "preprocess_checkpoint_qigen",
    "pack_model",
    "autogptq_post_init",
    "check_and_get_model_type",
    "simple_dispatch_model",
    "make_sure_no_tensor_in_meta_device"
]
