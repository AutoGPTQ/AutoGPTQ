from ._base import BaseGPTQForCausalLM


class MOSSGPTQ(BaseGPTQForCausalLM):
    layer_type = "MossBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]
