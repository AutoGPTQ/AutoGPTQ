from ._base import BaseGPTQForCausalLM

class MiniCPMOGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Qwen2DecoderLayer"
    layers_block_name = "llm.model.layers"
    outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm", "vpm", "apm", "tts", "audio_projection_layer", "resampler"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
__all__ = ["MiniCPMOGPTQForCausalLM"]