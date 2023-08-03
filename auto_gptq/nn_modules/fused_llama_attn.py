import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from ._fused_base import FusedBaseAttentionModule
from ..utils.import_utils import compare_pytorch_version, dynamically_import_QuantLinear

class FusedLlamaAttentionForQuantizedModel(FusedBaseAttentionModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        qkv_proj,
        o_proj,
        rotary_emb,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if self.config.pretraining_tp > 1:
            raise NotImplementedError(f"pretraining_tp of 2 or more is currently not supported.")
            
        if len(qkv_proj) == 1:
            self.qkv_mode = 'qkv'
            self.qkv_proj = qkv_proj[0]
        elif len(qkv_proj) == 2:
            self.qkv_mode = 'q,kv'
            self.q_proj = qkv_proj[0]
            self.kv_proj = qkv_proj[1]
        else:
            self.qkv_mode = 'q,k,v'
            self.q_proj = qkv_proj[0]
            self.k_proj = qkv_proj[1]
            self.v_proj = qkv_proj[2]
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()
        
        if self.qkv_mode == 'qkv':
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)
        elif self.qkv_mode == 'q,kv':
            query_states = self.q_proj(hidden_states)
            kv_states = self.kv_proj(hidden_states)
            key_states, value_states = torch.split(kv_states, self.hidden_size, dim=2)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        is_causal = past_key_value is None
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if compare_pytorch_version("v2.0.0", op="eq"):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None if is_causal else attention_mask,
                is_causal=is_causal
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @classmethod
    def inject_to_model(
        cls,
        model,
        use_triton=False,
        group_size=-1,
        use_cuda_fp16=True,
        desc_act=False,
        trainable=False,
        bits: int = 4,
        **kwargs
    ):
        """
        Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
        """
        QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits)

        for name, m in model.named_modules():
            if not isinstance(m, LlamaAttention):
                continue

            q_proj = m.q_proj
            k_proj = m.k_proj
            v_proj = m.v_proj

            if QuantLinear.QUANT_TYPE == "exllama" and desc_act:
                qkv_layers = [q_proj,k_proj,v_proj]
            elif m.num_heads == m.num_key_value_heads:
                qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
                qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
                scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
                if QuantLinear.QUANT_TYPE == "exllama":
                    g_idx = None
                else:
                    g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
                bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

                qlinear_args = (
                    q_proj.bits,
                    q_proj.group_size,
                    q_proj.infeatures,
                    q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures,
                    True if q_proj.bias is not None else False,
                )
                qlinear_kwargs = {"trainable": trainable}
                if (not desc_act or group_size == -1) and not use_triton:
                    qlinear_kwargs["use_cuda_fp16"] = use_cuda_fp16
                qkv_layer = QuantLinear(*qlinear_args, **qlinear_kwargs)
                qkv_layer.qweight = qweights
                qkv_layer.qzeros = qzeros
                qkv_layer.scales = scales
                qkv_layer.g_idx = g_idx
                qkv_layer.bias = bias
                qkv_layers = [qkv_layer]
            else:
                qweights = torch.cat([k_proj.qweight, v_proj.qweight], dim=1)
                qzeros = torch.cat([k_proj.qzeros, v_proj.qzeros], dim=1)
                scales = torch.cat([k_proj.scales, v_proj.scales], dim=1)
                if QuantLinear.QUANT_TYPE == "exllama":
                    g_idx = None
                else:
                    g_idx = torch.cat([k_proj.g_idx, v_proj.g_idx], dim=0)
                bias = torch.cat([k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

                qlinear_args = (
                    k_proj.bits,
                    k_proj.group_size,
                    k_proj.infeatures,
                    k_proj.outfeatures + v_proj.outfeatures,
                    True if q_proj.bias is not None else False,
                )
                qlinear_kwargs = {"trainable": trainable}
                if (not desc_act or group_size == -1) and not use_triton:
                    qlinear_kwargs["use_cuda_fp16"] = use_cuda_fp16
                kv_layer = QuantLinear(*qlinear_args, **qlinear_kwargs)
                kv_layer.qweight = qweights
                kv_layer.qzeros = qzeros
                kv_layer.scales = scales
                kv_layer.g_idx = g_idx
                kv_layer.bias = bias
                qkv_layers = [q_proj, kv_layer]
            attn = cls(m.config, qkv_layers, m.o_proj, m.rotary_emb)

            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name

            setattr(parent, child_name, attn)

__all__ = ["FusedLlamaAttentionForQuantizedModel"]
