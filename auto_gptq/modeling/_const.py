from packaging.version import parse as parse_version

from torch import device

from ..utils.import_utils import compare_transformers_version

CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = [
    "bloom",
    "gptj",
    "gpt2",
    "gpt_neox",
    "opt",
    "moss",
    "gpt_bigcode",
    "codegen",
    "RefinedWebModel",
    "RefinedWeb",
    "baichuan",
    "internlm",
    "qwen",
    "xverse",
    "deci_lm",
]
if compare_transformers_version("v4.28.0", op="ge"):
    SUPPORTED_MODELS.append("llama")
if compare_transformers_version("v4.33.0", op="ge"):
    SUPPORTED_MODELS.append("falcon")
if compare_transformers_version("v4.34.0", op="ge"):
    SUPPORTED_MODELS.append("mistral")
    SUPPORTED_MODELS.append("Yi")


EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS", "EXLLAMA_DEFAULT_MAX_INPUT_LENGTH"]
