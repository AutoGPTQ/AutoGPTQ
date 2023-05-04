from typing import Optional

from ._base import BaseQuantizeConfig, BaseGPTQForCausalLM
from ._utils import check_and_get_model_type
from .bloom import BloomGPTQForCausalLM
from .gpt_neox import GPTNeoXGPTQForCausalLM
from .gptj import GPTJGPTQForCausalLM
from .gpt2 import GPT2GPTQForCausalLM
from .llama import LlamaGPTQForCausalLM
from .moss import MOSSGPTQForCausalLM
from .opt import OPTGPTQForCausalLM
from inspect import signature

GPTQ_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomGPTQForCausalLM,
    "gpt_neox": GPTNeoXGPTQForCausalLM,
    "gptj": GPTJGPTQForCausalLM,
    "gpt2": GPT2GPTQForCausalLM,
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
        max_memory: Optional[dict] = None,
        **model_init_kwargs
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            max_memory=max_memory,
            **model_init_kwargs
        )

    @classmethod
    def from_quantized(
        cls,
        save_dir: str,
        device: str = "cpu",
        strict: bool = True,
        use_safetensors: bool = False,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        max_memory: Optional[dict] = None,
        device_map: Optional[str] = None,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        trust_remote_code: bool = False, 
        **kwargs
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(save_dir)
        quant_func = GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized
        keywords = {key: kwargs[key] for key in signature(quant_func).parameters if key in kwargs}
        return quant_func(
            save_dir=save_dir,
            device=device,
            strict=strict,
            use_safetensors=use_safetensors,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            max_memory=max_memory,
            device_map=device_map,
            quantize_config=quantize_config,
            model_basename=model_basename,
            trust_remote_code=trust_remote_code,
            **keywords
        )

__all__ = ["AutoGPTQForCausalLM"]
