from transformers import (
    AutoModelForCausalLM,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
)

from . import BaseGPTQForCausalLM
from ._const import SUPPORTED_MODELS


class Qwen2VLGPTQForConditionalGeneration(BaseGPTQForCausalLM):
    layer_type = "Qwen2VLDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm", "visual"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    # hack so one can prepare examples outside
    def _prepare_examples_for_quantization(self, examples, batch_size: int = 1):
        return examples


# hack to make sure Qwen2VLGPTQForConditionalGeneration.from_pretrained works properly
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)
SUPPORTED_MODELS.append(Qwen2VLConfig.model_type)


__all__ = ["Qwen2VLGPTQForConditionalGeneration"]
