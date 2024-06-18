from ._base import BaseGPTQForCausalLM


class ChatGLMForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GLMBlock"
    layers_block_name = "transformer.encoder.layers"
    outside_layer_modules = ["transformer.embedding.word_embeddings", "transformer.output_layer"]
    inside_layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
