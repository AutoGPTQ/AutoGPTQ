import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import xformers.ops as xop
from vllm import pos_encoding_ops as vllm_pos_encoding_ops
from xformers.ops.fmha.attn_bias import LowerTriangularMask, LowerTriangularMaskWithTensorBias


POTENTIAL_KV_CACHE_NAMES = (
    "past_key_value",
    "layer_past",
    "kv_cache"
)


def _try_to_get_kv_cache(**kwargs) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    kv_cache = None
    for name in POTENTIAL_KV_CACHE_NAMES:
        if name in kwargs:
            return kwargs[name]
    return kv_cache


def build_rope_cache(
    rotary_dim: int,
    max_position: int = 2048,
    base: int = 10000,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16
):  # TODO: support multiple scaling strategies
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
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(FusedAttention, self).__init__()

        self.qkv_proj = qkv_proj
        self.out_proj = out_proj

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads

        self.attn_dropout = attn_dropout if training else 0.0
        self.scale = scale

        self.attention_ops = attention_ops

        self.outputs_handler = outputs_handler

        self.resid_dropout = nn.Dropout(resid_dropout if training else 0.0)

    def _build_attn_bias(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[xop.AttentionBias]:
        return None

    def _attn(
        self,
        batch_size: int,
        seq_len: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[xop.AttentionBias] = None,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        q = q.view(batch_size, seq_len, self.num_query_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_value_heads, -1).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)

        present = None
        if use_cache:
            present = (k, v)

        attn_out = attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            attention_ops=self.attention_ops,
            attention_bias=attention_bias,
            p=self.attn_dropout,
            scale=self.scale
        ).view(batch_size, seq_len, -1)

        return attn_out, present

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]
        use_cache = kwargs.get("use_cache", False)
        kv_cache = _try_to_get_kv_cache(**kwargs)

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        attn_bias = self._build_attn_bias(hidden_states, attention_mask)
        attn_out, present = self._attn(
            bsz,
            seq_len,
            q,
            k,
            v,
            attn_bias,
            use_cache,
            kv_cache
        )

        out = self.out_proj(attn_out)
        out = self.resid_dropout(out)

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
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xop.AttentionOp] = None,
        outputs_handler: Optional[Callable] = None,
        training: bool = False,
    ):
        super(FusedAttentionWithRoPE, self).__init__(
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            scale=scale,
            attention_ops=attention_ops,
            outputs_handler=outputs_handler,
            training=training
        )

        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def _build_attn_bias(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[xop.AttentionBias]:
        return LowerTriangularMask()

    def _apply_rotary_embedding(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ):
        bsz, seq_len, hidden_size = query.shape

        if position_ids is not None:
            query = query.view(bsz * seq_len, -1)
            key = key.view(bsz * seq_len, -1)
            vllm_pos_encoding_ops.rotary_embedding_neox(
                position_ids.view(-1).to(query.device),
                query,
                key,
                hidden_size // self.num_query_heads,
                self.cos_sin_cache,
            )
            query = query.view(bsz, seq_len, -1)
            key = key.view(bsz, seq_len, -1)

        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]
        position_ids = kwargs.get("position_ids", None)
        use_cache = kwargs.get("use_cache", False)
        kv_cache = _try_to_get_kv_cache(**kwargs)

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        q, k = self._apply_rotary_embedding(q, k, position_ids)

        attn_bias = self._build_attn_bias(hidden_states, attention_mask) if kv_cache is None else None
        attn_out, present = self._attn(
            bsz,
            seq_len,
            q,
            k,
            v,
            attn_bias,
            use_cache,
            kv_cache
        )

        out = self.out_proj(attn_out)
        out = self.resid_dropout(out)

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
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
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
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            scale=scale,
            attention_ops=attention_ops,
            outputs_handler=outputs_handler,
            training=training
        )

        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

    def _build_attn_bias(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[xop.AttentionBias]:  # adopt from vllm
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
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        bsz, seq_len = hidden_states.shape[:2]
        use_cache = kwargs.get("use_cache", False)
        kv_cache = _try_to_get_kv_cache(**kwargs)

        q, k, v = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        attn_bias = self._build_attn_bias(hidden_states, attention_mask) if kv_cache is None else None
        attn_out, present = self._attn(
            bsz,
            seq_len,
            q,
            k,
            v,
            attn_bias,
            use_cache,
            kv_cache
        )

        out = self.out_proj(attn_out)
        out = self.resid_dropout(out)

        outputs = (out, present, None)
        if self.outputs_handler:
            outputs = self.outputs_handler(*outputs)

        return outputs
