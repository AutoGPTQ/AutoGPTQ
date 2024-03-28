from ._base import BaseGPTQForCausalLM


class DbrxGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DbrxBlock"
    layers_block_name = "transformer.blocks"
    outside_layer_modules = ["transformer.wte", "transformer.norm_f"]
    inside_layer_modules = [
        ["norm_attn_norm.attn.Wqkv"],
        ["norm_attn_norm.attn.o_proj"],
        ["ffn.experts.mlp.w1", "ffn.experts.mlp.v1"],
        ["ffn.experts.mlp.w2"],
    ]


__all__ = ["DbrxGPTQForCausalLM"]
