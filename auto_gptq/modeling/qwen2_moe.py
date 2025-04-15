from ._base import BaseGPTQForCausalLM


class Qwen2MoeGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Qwen2DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.shared_expert.up_proj", "mlp.shared_expert.gate_proj"],
        ["mlp.shared_expert.down_proj"],
        ["mlp.experts.{expert_idx}.up_proj", "mlp.experts.{expert_idx}.gate_proj"],
        ["mlp.experts.{expert_idx}.down_proj"],
    ]


__all__ = ["Qwen2MoeGPTQForCausalLM"]
