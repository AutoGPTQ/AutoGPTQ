from functools import wraps

import torch
from auto_gptq.modeling._base import BaseGPTQForCausalLM
from peft import get_peft_model, PeftConfig, PeftModel, TaskType, PeftType
from peft.import_utils import is_bnb_available
from peft.tuners.lora import Linear, Embedding

lora_embedding = (Embedding,)
lora_linear = (Linear,)
if is_bnb_available():
    from peft.tuners.lora import Linear8bitLt

    lora_linear += (Linear8bitLt,)


def get_gptq_peft_model(
    model: BaseGPTQForCausalLM,
    peft_config: PeftConfig = None,
    model_id: str = None,
    adapter_name: str = "default"
):
    if not peft_config.task_type == TaskType.CAUSAL_LM:
        raise TypeError("only support CAUSAL_LM task type.")

    try:
        if model_id is None:
            if not peft_config:
                raise ValueError("peft_config can't be None when model_id is None.")
            peft_model = get_peft_model(model.model, peft_config)
        else:
            peft_model = PeftModel.from_pretrained(model.model, model_id, adapter_name)
    except:
        raise NotImplementedError(f"gptq model not support {peft_config.peft_type.value} peft type yet.")

    if peft_config.peft_type == PeftType.LORA:
        for n, m in model.named_modules():
            if isinstance(m, lora_embedding + lora_linear):
                old_forward = m.forward

                @wraps(old_forward)
                def new_forward(*args, **kwargs):
                    args = [arg.type_as(m.weight.data) if isinstance(arg, torch.Tensor) else arg for arg in args]
                    return old_forward(*args, **kwargs)

                m.forward = new_forward

    if peft_config.peft_type == PeftType.ADALORA:
        pass

    return peft_model
