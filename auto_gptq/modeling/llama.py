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

    _fused_attention_module_type = None
    _fused_mlp_module_type = None

    @classmethod
    def get_fused_mlp_module(cls):
        if cls._fused_mlp_module_type is None:
            try:
                from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
                cls._fused_mlp_module_type = FusedLlamaMLPForQuantizedModel
            except ImportError:
                logger.error("Triton required for Fused MLP but not found.")
            except:
                raise
        return cls._fused_mlp_module_type

    @classmethod
    def get_fused_attention_module(cls):
        try:
            if cls._fused_attention_module_type is None:
                from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
                cls._fused_attention_module_type = FusedLlamaAttentionForQuantizedModel
        except ImportError:
            logger.error("Failed to import FusedLlamaAttentionForQuantizedModel")
        except:
            raise

        return cls._fused_attention_module_type

__all__ = ["LlamaGPTQForCausalLM"]
