from ._base import BaseQuantizeConfig
from bloom import *
from gpt_neox import *
from gptj import *
from llama import *
from opt import *

GPTQ_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomGPTQForCausalLM,
    "gpt_neox": GPTNeoXGPTQForCausalLM,
    "gptj": GPTJGPTQForCausalLM,
    "llama": LlamaGPTQForCausalLM,
    "opt": OPTGPTQForCausalLM
}
