from auto_gptq.modeling import BaseGPTQForCausalLM


class MPTGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "MPTBlock"
    layers_block_name = "transformer.blocks"
    outside_layer_modules = [
        "transformer.wte",  "transformer.norm_f"
    ]

    inside_layer_modules = [
        ["attn.Wqkv"],
        ["attn.out_proj"],
        ["ffn.up_proj"],
        ["ffn.down_proj"]
    ]

__all__ = ["MPTGPTQForCausalLM"]