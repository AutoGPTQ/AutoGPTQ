import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import xformers.ops as xop
from vllm import pos_encoding_ops as vllm_pos_encoding_ops
from xformers.ops.fmha.attn_bias import LowerTriangularMask, LowerTriangularMaskWithTensorBias


def build_rope_cache(
    rotary_dim: int,
    max_position: int = 2048,
    base: int = 10000,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16
):
    inv_freq = (1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)))
    t = torch.arange(max_position, device=device, dtype=dtype)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)

    return cache


def build_alibi_slopes(
    num_heads: int,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16
):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    slopes = slopes.to(dtype)

    return slopes


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_ops: Optional[xop.AttentionOp] = (xop.fmha.flash.FwOp(), None),
    attention_bias: Optional[xop.AttentionBias] = None,
    p: float = 0.0,
    scale: Optional[float] = None
):
    if value.shape[2] != query.shape[2]:
        # MQA expand
        if value.shape[2] == 1:
            pass  # TODO
        # GQA reshape
        else:
            original_shape = value.shape
            pass  # TODO

    return xop.memory_efficient_attention(
        query=query,
        key=key,
        value=value,
        attn_bias=attention_bias,
        p=p,
        scale=scale,
        op=attention_ops
    )


class FusedAttention(nn.Module):
    def __init__(
        self,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        attention_bias: Optional[xop.AttentionBias] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(FusedAttention, self).__init__()

        self.qkv_proj = qkv_proj
        self.out_proj = out_proj

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads

        self.dropout = dropout if training else 0.0
        self.scale = scale

        self.attention_ops = attention_ops
        self.attention_bias = attention_bias

        self.outputs_handler = outputs_handler

    def _attn(
        self,
        batch_size: int,
        seq_len: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[xop.AttentionBias] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        **kwargs
    ):
        use_cache = kwargs.get("use_cache", False)
        kv_cache = kwargs.get("layer_past", kwargs.get("past_key_value", kwargs.get("kv_cache", None)))

        q = q.view(batch_size, seq_len, self.num_query_heads, -1)
        k = k.view(batch_size, seq_len, self.num_key_heads, -1)
        v = v.view(batch_size, seq_len, self.num_value_heads, -1)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=1)
            v = torch.cat((v_cache, v), dim=1)

        present = None
        if use_cache:
            present = (k, v)

        attn_ops = attention_ops if attention_ops is not None else self.attention_ops
        attn_bias = attention_bias if attention_bias is not None else self.attention_bias
        attn_out = attention(
            query=q,
            key=k,
            value=v,
            attention_ops=attn_ops,
            attention_bias=attn_bias if kv_cache is None else None,
            p=self.dropout,
            scale=self.scale
        ).view(batch_size, seq_len, -1)

        return attn_out, present

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        attn_out, present = self._attn(bsz, seq_len, q, k, v, **kwargs)

        out = self.out_proj(attn_out)

        outputs = (out, present, None)
        if self.outputs_handler:
            outputs = self.outputs_handler(*outputs)

        return outputs


class FusedAttentionWithRoPE(FusedAttention):
    def __init__(
        self,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        cos_sin_cache: torch.Tensor,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        attention_bias: Optional[xop.AttentionBias] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(FusedAttentionWithRoPE, self).__init__(
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            dropout=dropout,
            scale=scale,
            attention_ops=attention_ops,
            attention_bias=attention_bias,
            outputs_handler=outputs_handler,
            training=training
        )

        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ):
        position_ids = kwargs.get("position_ids", None)

        bsz, seq_len = hidden_states.shape[:2]

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)
        q = q.view(bsz * seq_len, -1)
        k = k.view(bsz * seq_len, -1)

        if position_ids is not None:
            vllm_pos_encoding_ops.rotary_embedding_neox(
                position_ids.view(-1).to(q.device),
                q,
                k,
                q.shape[-1] // self.num_query_heads,
                self.cos_sin_cache,
            )

        attn_out, present = self._attn(bsz, seq_len, q, k, v, **kwargs)

        out = self.out_proj(attn_out)

        outputs = (out, present, None)
        if self.outputs_handler:
            outputs = self.outputs_handler(*outputs)

        return outputs


class FusedAttentionWithALiBi(FusedAttention):
    def __init__(
        self,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        alibi_slopes: torch.Tensor,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(FusedAttentionWithALiBi, self).__init__(
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            dropout=dropout,
            scale=scale,
            attention_ops=attention_ops,
            outputs_handler=outputs_handler,
            training=training
        )

        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

    def _build_attention_bias(self, hidden_states: torch.Tensor):  # adopt from vllm
        bsz, seq_len = hidden_states.shape[:2]

        bias = torch.arange(seq_len)
        bias = bias[None, :] - bias[:, None]
        bias = bias.to(hidden_states.device)

        # When using custom attention bias, xformers requires the bias to
        # be sliced from a tensor whose length is a multiple of 8.
        padded_len = (seq_len + 7) // 8 * 8
        bias = torch.empty(
            self.num_query_heads,
            padded_len,
            padded_len,
            device=self.alibi_slopes.device,
        )[:, :seq_len, :seq_len].copy_(bias)
        bias.mul_(self.alibi_slopes[:, None, None])
        bias = LowerTriangularMaskWithTensorBias(bias.unsqueeze(0).repeat(bsz, 1, 1, 1))

        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        kv_cache = kwargs.get("layer_past", kwargs.get("past_key_value", kwargs.get("kv_cache", None)))
        attn_bias = self._build_attention_bias(hidden_states) if kv_cache is None else None
        attn_out, present = self._attn(bsz, seq_len, q, k, v, attention_bias=attn_bias, **kwargs)

        out = self.out_proj(attn_out)

        outputs = (out, present, None)
        if self.outputs_handler:
            outputs = self.outputs_handler(*outputs)

        return outputs
