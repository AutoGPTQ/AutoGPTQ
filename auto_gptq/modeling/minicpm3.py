from ._base import BaseGPTQForCausalLM
class MiniCPM3GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "MiniCPM3DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = [
        "model.embed_tokens",
    ]
    inside_layer_modules = [
        ["self_attn.q_a_proj","self_attn.kv_a_proj_with_mqa"],
        ["self_attn.q_b_proj","self_attn.kv_b_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj","mlp.up_proj"],
        ["mlp.down_proj"],
    ]
__all__ = ["MiniCPM3GPTQForCausalLM"]