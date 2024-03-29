from ._base import BaseGPTQForCausalLM


class DbrxGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DbrxBlock"
    layers_block_name = "transformer.blocks"
    outside_layer_modules = ["transformer.wte", "transformer.norm_f"]
    inside_layer_modules = [
        ["norm_attn_norm.norm_1"],
        ["norm_attn_norm.norm_2"],
        ["norm_attn_norm.attn.Wqkv"],
        ["norm_attn_norm.attn.out_proj"],
        ["ffn.router.layer"],
        ["ffn.experts.mlp"],
    ]


__all__ = ["DbrxGPTQForCausalLM"]
