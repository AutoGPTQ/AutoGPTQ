from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import xformers.ops as xop
from torch.cuda import empty_cache
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from xformers.ops.fmha import AttentionOp

from ._base import *
from ..nn_modules.fused_modules.linear import FusedGeneralQuantLinear
from ..nn_modules.fused_modules.attention import FusedAttention
from ..nn_modules.fused_modules.mlp import FusedMLP


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dtype = x.dtype
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp).to(dtype), torch.cos(sinusoid_inp).to(dtype)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


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

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = key.shape[:2]

        query = query.view(bsz, seq_len, self.num_query_heads, -1)
        key = key.view(bsz, seq_len, self.num_key_heads, -1)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        return query, key

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
            q, k = self._apply_rotary(q, k, layer_past)

        attn_bias = self._build_attn_bias(hidden_states, attention_mask)
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
            new_qkv_proj = FusedGeneralQuantLinear.fuse(
                old_attn.q_proj,
                old_attn.k_proj,
                old_attn.v_proj
            )
            new_out_proj = FusedGeneralQuantLinear(old_attn.out_proj)
            new_attn = GPTJFusedAttention(
                qkv_proj=new_qkv_proj,
                out_proj=new_out_proj,
                embed_positions=old_attn.embed_positions.to(old_attn.q_proj.qweight.data.device),
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
