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
from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel

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

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel

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
        model = super(LlamaGPTQForCausalLM, cls).from_quantized(
            save_dir=save_dir,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            strict=strict,
            use_triton=use_triton,
            inject_fused_attention=inject_fused_attention,
            inject_fused_mlp=inject_fused_mlp,
            use_cuda_fp16=use_cuda_fp16,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            **kwargs
        )

        if use_triton and warmup_triton and inject_fused_mlp:
            from ..nn_modules.fused_llama_mlp import autotune_warmup_fused
            autotune_warmup_fused(model, seqlen=model.seqlen)
            
        return model


__all__ = ["LlamaGPTQForCausalLM"]
