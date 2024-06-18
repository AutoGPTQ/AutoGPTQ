from ._base import BaseGPTQForCausalLM


class MPTGPTQ(BaseGPTQForCausalLM):
    layer_type = "MPTBlock"
    layers_block_name = "transformer.blocks"
    outside_layer_modules = ["transformer.wte", "transformer.norm_f"]

    inside_layer_modules = [["attn.Wqkv"], ["attn.out_proj"], ["ffn.up_proj"], ["ffn.down_proj"]]
