from ._base import BaseGPTQForCausalLM


class TeleChat2GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "TelechatBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.word_embeddings", "transformer.ln_f"]

    """
    If other frameworks are used for inference (such as VLLM), 
    it is best not to quantify QKV due to the organization of 
    key value weights in the Telechat model
    """   
    inside_layer_modules = [
        ["self_attention.dense"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

__all__ = ["TeleChat2GPTQForCausalLM"]
