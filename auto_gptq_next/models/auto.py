from inspect import signature
from typing import Dict, Optional, Union

from ._base import BaseGPTQForCausalLM, QuantizeConfig
from ._utils import check_and_get_model_type
from .baichuan import BaiChuanGPTQ
from .bloom import BloomGPTQ
from .chatglm import ChatGLM
from .codegen import CodeGenGPTQ
from .cohere import CohereGPTQ
from .decilm import DeciLMGPTQ
from .gemma import GemmaGPTQ
from .gpt2 import GPT2GPTQ
from .gpt_bigcode import GPTBigCodeGPTQ
from .gpt_neox import GPTNeoXGPTQ
from .gptj import GPTJGPTQ
from .internlm import InternLMGPTQ
from .llama import LlamaGPTQ
from .longllama import LongLlamaGPTQ
from .mistral import MistralGPTQ
from .mixtral import MixtralGPTQ
from .moss import MOSSGPTQ
from .mpt import MPTGPTQ
from .opt import OPTGPTQ
from .phi import PhiGPTQ
from .qwen import QwenGPTQ
from .qwen2 import Qwen2GPTQ
from .rw import RWGPTQ
from .stablelmepoch import StableLMEpochGPTQ
from .starcoder2 import Starcoder2GPTQ
from .xverse import XverseGPTQ
from .yi import YiGPTQ

MODEL_MAP = {
    "bloom": BloomGPTQ,
    "gpt_neox": GPTNeoXGPTQ,
    "gptj": GPTJGPTQ,
    "gpt2": GPT2GPTQ,
    "llama": LlamaGPTQ,
    "opt": OPTGPTQ,
    "moss": MOSSGPTQ,
    "chatglm": ChatGLM,
    "gpt_bigcode": GPTBigCodeGPTQ,
    "codegen": CodeGenGPTQ,
    "cohere": CohereGPTQ,
    "RefinedWebModel": RWGPTQ,
    "RefinedWeb": RWGPTQ,
    "falcon": RWGPTQ,
    "baichuan": BaiChuanGPTQ,
    "internlm": InternLMGPTQ,
    "qwen": QwenGPTQ,
    "mistral": MistralGPTQ,
    "Yi": YiGPTQ,
    "xverse": XverseGPTQ,
    "deci": DeciLMGPTQ,
    "stablelm_epoch": StableLMEpochGPTQ,
    "starcoder2": Starcoder2GPTQ,
    "mixtral": MixtralGPTQ,
    "qwen2": Qwen2GPTQ,
    "longllama": LongLlamaGPTQ,
    "gemma": GemmaGPTQ,
    "phi": PhiGPTQ,
    "mpt": MPTGPTQ,
}


class AutoGPTQNext:
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
        quantize_config: QuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path, trust_remote_code)
        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[QuantizeConfig | Dict] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        disable_exllama: Optional[bool] = None,
        disable_exllamav2: bool = False,
        use_marlin: bool = False,
        **kwargs,
    ) -> BaseGPTQForCausalLM:
        # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
        if disable_exllama is None:
            if disable_exllamav2:
                disable_exllama = False
            else:
                disable_exllama = True

        model_type = check_and_get_model_type(model_name_or_path, trust_remote_code)
        quant_func = MODEL_MAP[model_type].from_quantized
        # A static list of kwargs needed for huggingface_hub
        huggingface_kwargs = [
            "cache_dir",
            "force_download",
            "proxies",
            "resume_download",
            "local_files_only",
            "use_auth_token",
            "revision",
            "subfolder",
            "_raise_exceptions_for_missing_entries",
            "_commit_hash",
        ]
        # TODO: do we need this filtering of kwargs? @PanQiWei is there a reason we can't just pass all kwargs?
        keywords = {
            key: kwargs[key]
            for key in list(signature(quant_func).parameters.keys()) + huggingface_kwargs
            if key in kwargs
        }
        return quant_func(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            disable_exllama=disable_exllama,
            disable_exllamav2=disable_exllamav2,
            use_marlin=use_marlin,
            **keywords,
        )

