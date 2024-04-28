from logging import getLogger

from ..utils.import_utils import compare_transformers_version
from ._base import BaseGPTQForCausalLM


logger = getLogger(__name__)


class StableLMEpochGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


__all__ = ["StableLMEpochGPTQForCausalLM"]
