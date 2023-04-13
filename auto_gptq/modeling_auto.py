from transformers import AutoConfig

from .modeling import BaseQuantizeConfig, GPTQ_CAUSAL_LM_MODEL_MAP
from .modeling._const import SUPPORTED_MODELS


def check_and_get_model_type(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


class AutoGPTQModelForCausalLM:
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
        bf16: bool = False,
        **model_init_kwargs
    ):
        model_type = check_and_get_model_type(pretrained_model_name_or_path)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            bf16=bf16,
            **model_init_kwargs
        )

    @classmethod
    def from_quantized(
        cls,
        save_dir: str,
        device: str = "cpu",
        use_safetensors: bool = False
    ):
        model_type = check_and_get_model_type(save_dir)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            save_dir=save_dir,
            device=device,
            use_safetensors=use_safetensors
        )


__all__ = ["AutoGPTQModelForCausalLM"]
