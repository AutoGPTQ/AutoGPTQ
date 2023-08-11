from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import xformers.ops as xop
from torch.cuda import empty_cache
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.gptj.modeling_gptj import apply_rotary_pos_emb
from xformers.ops.fmha import AttentionOp

from ._base import *
from ..nn_modules.fused_modules.linear import FusedGeneralQuantLinear
from ..nn_modules.fused_modules.attention import FusedAttention
from ..nn_modules.fused_modules.mlp import FusedMLP


class GPTJFusedAttention(FusedAttention):
    def __init__(
        self,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        embed_positions: torch.Tensor,
        rotary_dim: Optional[int],
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(GPTJFusedAttention, self).__init__(
            qkv_proj,
            out_proj,
            num_query_heads,
            num_key_heads,
            num_value_heads,
            attn_dropout,
            resid_dropout,
            scale,
            attention_ops,
            outputs_handler,
            training
        )
        self.embed_positions = embed_positions
        self.rotary_dim = rotary_dim

    def _get_embed_positions(self, position_ids: torch.Tensor):
        return self.embed_positions.repeat(position_ids.shape[0], 1, 1)

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = key.shape[:2]

        dtype = query.dtype
        query = query.view(bsz, seq_len, self.num_query_heads, -1).to(dtype=torch.float)
        key = key.view(bsz, seq_len, self.num_key_heads, -1).to(dtype=torch.float)

        embed_positions = self._get_embed_positions(position_ids)

        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        return query.view(bsz, seq_len, -1).to(dtype=dtype), key.view(bsz, seq_len, -1).to(dtype=dtype)

    def _build_attn_bias(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[xop.AttentionBias]:
        return xop.LowerTriangularMask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        if position_ids is not None:
            q, k = self._apply_rotary(q, k, position_ids)

        attn_bias = self._build_attn_bias(hidden_states, attention_mask) if layer_past is None else None
        attn_out, present = self._attn(
            bsz,
            seq_len,
            q,
            k,
            v,
            attn_bias,
            use_cache,
            layer_past
        )

        out = self.out_proj(attn_out)
        out = self.resid_dropout(out)

        outputs = (out, present, None)
        if self.outputs_handler:
            outputs = self.outputs_handler(*outputs)

        return outputs


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
        layers = model.transformer.h

        for layer in layers:
            old_attn = layer.attn
            device = old_attn.q_proj.qweight.data.device
            new_qkv_proj = FusedGeneralQuantLinear.fuse(
                old_attn.q_proj,
                old_attn.k_proj,
                old_attn.v_proj
            )
            new_out_proj = FusedGeneralQuantLinear(old_attn.out_proj)
            new_attn = GPTJFusedAttention(
                qkv_proj=new_qkv_proj,
                out_proj=new_out_proj,
                embed_positions=old_attn.embed_positions.to(device),
                rotary_dim=old_attn.rotary_dim,
                num_query_heads=num_heads,
                num_key_heads=num_heads,
                num_value_heads=num_heads,
                attn_dropout=model_config.attn_pdrop,
                resid_dropout=model_config.resid_pdrop,
                scale=scale,
                attention_ops=attn_op,
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
        layers = model.transformer.h

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
