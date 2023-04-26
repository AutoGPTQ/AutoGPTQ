from typing import Optional

from ._base import BaseQuantizeConfig, BaseGPTQForCausalLM
from ._utils import check_and_get_model_type
from .bloom import BloomGPTQForCausalLM
from .gpt_neox import GPTNeoXGPTQForCausalLM
from .gptj import GPTJGPTQForCausalLM
from .llama import LlamaGPTQForCausalLM
from .moss import MOSSGPTQForCausalLM
from .opt import OPTGPTQForCausalLM


GPTQ_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomGPTQForCausalLM,
    "gpt_neox": GPTNeoXGPTQForCausalLM,
    "gptj": GPTJGPTQForCausalLM,
    "llama": LlamaGPTQForCausalLM,
    "opt": OPTGPTQForCausalLM,
    "moss": MOSSGPTQForCausalLM
}


class AutoGPTQForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoGPTQModelForCausalLM is designed to be instantiated\n"
            "using `AutoGPTQModelForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoGPTQModelForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        max_memory_per_gpu: Optional[int] = None,
        **model_init_kwargs
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            max_memory_per_gpu=max_memory_per_gpu,
            **model_init_kwargs
        )

    @classmethod
    def from_quantized(
        cls,
        save_dir: str,
        device: str = "cpu",
        use_safetensors: bool = False,
        use_triton: bool = False
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(save_dir)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            save_dir=save_dir,
            device=device,
            use_safetensors=use_safetensors,
            use_triton=use_triton
        )


__all__ = ["AutoGPTQForCausalLM"]
