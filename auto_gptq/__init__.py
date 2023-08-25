__version__ = "0.4.1"
from .modeling import BaseQuantizeConfig
from .modeling import AutoGPTQForCausalLM
from .utils.peft_utils import get_gptq_peft_model
from .utils.exllama_utils import exllama_set_max_input_length