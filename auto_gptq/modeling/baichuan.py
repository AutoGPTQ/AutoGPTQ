from copy import deepcopy
from typing import Optional

import xformers.ops as xop
from torch.cuda import empty_cache
from transformers import PreTrainedModel
from xformers.ops.fmha import AttentionOp

from ._base import *
from ..nn_modules.fused_modules.attention import build_rope_cache, FusedAttentionWithRoPE
from ..nn_modules.fused_modules.linear import FusedGeneralQuantLinear
from ..nn_modules.fused_modules.mlp import FusedGatedMLP


class BaiChuanFusedAttentionWithRope(FusedAttentionWithRoPE):
    pass


class BaiChuanGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.W_pack"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    @staticmethod
    def _fuse_attention(
        model: PreTrainedModel,
        attn_op: Optional[AttentionOp] = None,
        trainable: bool = False
    ) -> None:
        model_config = model.config
        num_heads = model_config.num_attention_heads
        scale = (model_config.hidden_size // num_heads) ** -0.5
        layers = model.model.layers

        rope_cache = build_rope_cache(
            rotary_dim=model_config.hidden_size // num_heads,
            max_position=model_config.max_position_embeddings,
            device=model.device,
            dtype=model.dtype
        )

        for layer in layers:
            old_attn = layer.self_attn
            attn_device = old_attn.W_pack.qweight.data.device
            new_qkv_proj = FusedGeneralQuantLinear(old_attn.W_pack)
            new_out_proj = FusedGeneralQuantLinear(old_attn.o_proj)
            new_attn = BaiChuanFusedAttentionWithRope(
                qkv_proj=new_qkv_proj,
                out_proj=new_out_proj,
                cos_sin_cache=rope_cache if attn_device == model.device else deepcopy(rope_cache).to(attn_device),
                num_query_heads=num_heads,
                num_key_heads=num_heads,
                num_value_heads=num_heads,
                attn_dropout=0.0,
                resid_dropout=0.0,
                scale=scale,
                attention_ops=attn_op,
                outputs_handler=(lambda x, y, z: (x, z, y)),
                training=trainable
            )

            layer.self_attn = new_attn

            del old_attn

        empty_cache()


__all__ = ["BaiChuanGPTQForCausalLM"]
