from ._base import BaseGPTQForCausalLM


class PhiGPTQForCausalLM(BaseGPTQForCausalLM):
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


class Phi3GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Phi3DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]


class PhiMoEGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "PhiMoEDecoderLayer"
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
            "block_sparse_moe.experts.8.w1",
            "block_sparse_moe.experts.9.w1",
            "block_sparse_moe.experts.10.w1",
            "block_sparse_moe.experts.11.w1",
            "block_sparse_moe.experts.12.w1",
            "block_sparse_moe.experts.13.w1",
            "block_sparse_moe.experts.14.w1",
            "block_sparse_moe.experts.15.w1",
            "block_sparse_moe.experts.8.w3",
            "block_sparse_moe.experts.9.w3",
            "block_sparse_moe.experts.10.w3",
            "block_sparse_moe.experts.11.w3",
            "block_sparse_moe.experts.12.w3",
            "block_sparse_moe.experts.13.w3",
            "block_sparse_moe.experts.14.w3",
            "block_sparse_moe.experts.15.w3",
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
            "block_sparse_moe.experts.8.w2",
            "block_sparse_moe.experts.9.w2",
            "block_sparse_moe.experts.10.w2",
            "block_sparse_moe.experts.11.w2",
            "block_sparse_moe.experts.12.w2",
            "block_sparse_moe.experts.13.w2",
            "block_sparse_moe.experts.14.w2",
            "block_sparse_moe.experts.15.w2"
        ],
    ]


__all__ = ["PhiGPTQForCausalLM", "Phi3GPTQForCausalLM", "PhiMoEGPTQForCausalLM"]

