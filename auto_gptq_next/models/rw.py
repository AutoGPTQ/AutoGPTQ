from ._base import BaseGPTQForCausalLM


class RWGPTQ(BaseGPTQForCausalLM):
    layer_type = "DecoderLayer"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.word_embeddings", "transformer.ln_f"]
    inside_layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
