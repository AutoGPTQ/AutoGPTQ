from logging import getLogger
from os.path import join, isfile
from typing import Optional, Union

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from ._const import *
from ._utils import *

from ._base import *

logger = getLogger(__name__)


class LlamaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    @classmethod
    def from_quantized(
        cls,
        save_dir: str,
        device_map: Optional[str] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        strict: bool = True,
        use_triton: bool = False,
        inject_fused_attention: bool = False,
        inject_fused_mlp: bool = False,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = False,
        trust_remote_code: bool = False,
        warmup_triton: bool = True,
        **kwargs
    ):
        """load quantized model from local disk"""
        if use_triton:
            if inject_fused_mlp:
                from ..nn_modules.fused_mlp_triton import make_fused_mlp
            if warmup_triton:
                from ..nn_modules.qlinear_triton import autotune_warmup_linear
                from ..nn_modules.fused_mlp_triton import autotune_warmup_fused
        if inject_fused_attention:
            from ..nn_modules.fused_llama_attn import make_quant_attn

        if device is None and not device_map and not max_memory:
            device_map = "auto"
        if device is not None:
            device = torch.device(device)
            if not max_memory and not device_map:
                device_map = {"": device.index if device.type == "cuda" else device.type}

        # prepare configs and file names
        config = AutoConfig.from_pretrained(save_dir, trust_remote_code=trust_remote_code)
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(save_dir)

        if model_basename is None:
            model_basename = f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g"

        model_save_name = join(save_dir, model_basename)

        if use_safetensors:
            model_save_name += ".safetensors"
        else:
            model_save_name += ".bin"

        if not isfile(model_save_name):
           raise FileNotFoundError(f"Could not find model at {model_save_name}")

        # inject layers
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False
        if strict:
            with accelerate.init_empty_weights():
                torch.set_default_dtype(torch.half)
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
                torch.set_default_dtype(torch.float)
        else:
            torch.set_default_dtype(torch.half)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
            torch.set_default_dtype(torch.float)
        layers = find_layers(model)
        ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                del layers[name]
        if strict:
            with accelerate.init_empty_weights():
                make_quant(
                    model,
                    layers,
                    quantize_config.bits,
                    quantize_config.group_size,
                    use_triton=use_triton,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act
                )
            model.tie_weights()
        else:
            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act
            )

        # load checkpoint and dispatch
        if strict:
            model = accelerate.load_checkpoint_and_dispatch(
                model,
                model_save_name,
                device_map,
                max_memory,
                no_split_module_classes=[cls.layer_type]
            )
        else:
            if use_safetensors:
                from safetensors.torch import load_file as safe_load
                model.load_state_dict(safe_load(model_save_name), strict=False)
            else:
                model.load_state_dict(torch.load(model_save_name), strict=False)
            if device_map == "auto":
                device_map = accelerate.infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type]
                )
            model = accelerate.dispatch_model(model, device_map)

        # inject fused layers
        if inject_fused_attention:
            make_quant_attn(
                model,
                use_triton=use_triton,
                group_size=quantize_config.group_size,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act
            )
        if use_triton and inject_fused_mlp:
            make_fused_mlp(model)

        # set seqlen
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        model.eval()

        # warmup triton
        if use_triton and warmup_triton:
            autotune_warmup_linear(model, seqlen=model.seqlen)
            if inject_fused_mlp:
                autotune_warmup_fused(model, seqlen=model.seqlen)
            
        return cls(model, True, quantize_config)


__all__ = ["LlamaGPTQForCausalLM"]
