from ._base import BaseGPTQForCausalLM


class Starcoder2GPTQ(BaseGPTQForCausalLM):
    layer_type = "Starcoder2DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ]
