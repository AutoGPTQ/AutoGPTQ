from ._base import *


class MixtralGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "MixtralDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [
            "block_sparse_moe.experts.0.w1",
            "block_sparse_moe.experts.1.w1",
            "block_sparse_moe.experts.2.w1",
            "block_sparse_moe.experts.3.w1",
            "block_sparse_moe.experts.4.w1",
            "block_sparse_moe.experts.5.w1",
            "block_sparse_moe.experts.6.w1",
            "block_sparse_moe.experts.7.w1",
            "block_sparse_moe.experts.0.w3",
            "block_sparse_moe.experts.1.w3",
            "block_sparse_moe.experts.2.w3",
            "block_sparse_moe.experts.3.w3",
            "block_sparse_moe.experts.4.w3",
            "block_sparse_moe.experts.5.w3",
            "block_sparse_moe.experts.6.w3",
            "block_sparse_moe.experts.7.w3",
         ],
        [
            "block_sparse_moe.experts.0.w2",
            "block_sparse_moe.experts.1.w2",
            "block_sparse_moe.experts.2.w2",
            "block_sparse_moe.experts.3.w2",
            "block_sparse_moe.experts.4.w2",
            "block_sparse_moe.experts.5.w2",
            "block_sparse_moe.experts.6.w2",
            "block_sparse_moe.experts.7.w2",
        ]
    ]


__all__ = ["MixtralGPTQForCausalLM"]