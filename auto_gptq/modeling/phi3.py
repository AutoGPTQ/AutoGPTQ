from ._base import BaseGPTQForCausalLM


class Phi3GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Phi3DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "embed_dropout", "model.norm"]
    inside_layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]


__all__ = ["Phi3GPTQForCausalLM"]
