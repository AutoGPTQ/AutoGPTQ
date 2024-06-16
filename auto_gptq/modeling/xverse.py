from ._base import BaseGPTQForCausalLM


class XverseGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "XverseDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


__all__ = ["XverseGPTQForCausalLM"]
