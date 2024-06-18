from ._base import BaseGPTQForCausalLM


class PhiGPTQ(BaseGPTQForCausalLM):
    layer_type = "PhiDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.final_layernorm"]
    inside_layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.dense"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]
