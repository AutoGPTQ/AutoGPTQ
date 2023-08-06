from copy import deepcopy
from typing import Optional

from torch.cuda import empty_cache
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from xformers.ops.fmha import AttentionOp, MemoryEfficientAttentionCutlassOp, LowerTriangularMask
from xformers.ops.fmha.cutlass import FwOp as CutlassFwOp

from ._base import *
from ._utils import get_module_by_name_prefix
from ..nn_modules.fused_modules.linear import FusedGeneralQuantLinear
from ..nn_modules.fused_modules.attention import build_rope_cache, FusedAttentionWithRoPE
from ..nn_modules.fused_modules.mlp import FusedMLP


class GPTJGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GPTJBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.k_proj", "attn.v_proj", "attn.q_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"]
    ]

    @staticmethod
    def _fuse_attention(
        model: PreTrainedModel,
        attn_op: Optional[AttentionOp] = None,
        trainable: bool = False
    ) -> None:
        model_config = model.config
        num_heads = model_config.n_head
        scale = (model_config.hidden_size // num_heads) ** -0.5
        layers = get_module_by_name_prefix(model, "transformer.h")

        rope_cache = build_rope_cache(
            rotary_dim=model_config.rotary_dim or model_config.hidden_size,
            max_position=model_config.max_position_embeddings,
            device=model.device,
            dtype=model.dtype
        )

        for layer in layers:
            old_attn = layer.attn
            attn_device = old_attn.q_proj.qweight.data.device
            new_qkv_proj = FusedGeneralQuantLinear.fuse(
                old_attn.q_proj,
                old_attn.k_proj,
                old_attn.v_proj
            )
            new_out_proj = FusedGeneralQuantLinear(old_attn.out_proj)
            new_attn = FusedAttentionWithRoPE(
                qkv_proj=new_qkv_proj,
                out_proj=new_out_proj,
                cos_sin_cache=rope_cache if attn_device == model.device else deepcopy(rope_cache).to(attn_device),
                num_query_heads=num_heads,
                num_key_heads=num_heads,
                num_value_heads=num_heads,
                dropout=model_config.attn_pdrop,
                scale=scale,
                attention_ops=attn_op,
                attention_bias=LowerTriangularMask(),
                outputs_handler=None,
                training=trainable
            )

            layer.attn = new_attn

            del old_attn

        empty_cache()

    @staticmethod
    def _fuse_mlp(
        model: PreTrainedModel,
        trainable: bool = False
    ) -> None:
        model_config = model.config
        act = ACT2FN[model_config.activation_function]
        out_dropout = model_config.resid_pdrop
        layers = get_module_by_name_prefix(model, "transformer.h")

        for layer in layers:
            old_mlp = layer.mlp
            new_mlp = FusedMLP(
                input_proj=FusedGeneralQuantLinear(old_mlp.fc_in),
                out_proj=FusedGeneralQuantLinear(old_mlp.fc_out),
                activation=act,
                in_dropout=0.0,
                out_dropout=out_dropout,
                training=trainable,
                residual=False
            )

            layer.mlp = new_mlp

            del old_mlp

        empty_cache()


__all__ = ["GPTJGPTQForCausalLM"]
