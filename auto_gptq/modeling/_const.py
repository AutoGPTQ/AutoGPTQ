from torch import device

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
    "llama",
    "qwen",
]

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS", "EXLLAMA_DEFAULT_MAX_INPUT_LENGTH"]
