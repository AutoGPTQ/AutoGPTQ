from ._base import BaseGPTQForCausalLM


class MiniCPMGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "MiniCPMDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = [
        "model.embed_tokens",
    ]
    inside_layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj","mlp.down_proj"],
        ["mlp.c_proj"],
    ]


__all__ = ["MiniCPMGPTQForCausalLM"]
