from ._base import BaseGPTQForCausalLM


class InternLMGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "InternLMDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm", "vit", "vision_proj", "model.tok_embeddings", "output"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
        ["attention.wqkv.linear"],  # InternLMXComposer2
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


__all__ = ["InternLMGPTQForCausalLM"]
