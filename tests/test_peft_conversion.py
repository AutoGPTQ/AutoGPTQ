import math
from unittest import TestCase

import torch.cuda.amp
from peft import TaskType
from peft.peft_model import PeftModelForCausalLM
from torch.optim import Adam
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.utils.peft_utils import (
    GPTQAdaLoraConfig,
    GPTQLoraConfig,
    GPTQLoraLinear,
    GPTQSVDLinear,
    get_gptq_peft_model,
)


MODEL_NAME = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"


class TestPeftConversion(TestCase):
    def check_model_trainable(self, model_lora: PeftModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        batch = tokenizer("Hello, world", return_tensors="pt")
        batch = {key: value.to(model_lora.device) for key, value in batch.items()}
        batch["labels"] = batch["input_ids"]
        batch["attention_mask"] = batch["attention_mask"].float()
        batch["attention_mask"].requires_grad = True
        model_lora.gradient_checkpointing_enable()
        optimizer = Adam(model_lora.parameters(), lr=1e-4)
        model_lora.train()
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model_lora(**batch).loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertTrue(losses[0] > losses[-1])
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertTrue(not any(math.isnan(loss) for loss in losses))

    def test_lora_conversion(self):
        model = AutoGPTQForCausalLM.from_quantized(
            MODEL_NAME,
            use_triton=False,
            warmup_triton=False,
            trainable=True,
            inject_fused_attention=True,
            inject_fused_mlp=False,
            use_safetensors=True,
        )
        peft_config = GPTQLoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["qkv_proj"],
        )
        model_lora = get_gptq_peft_model(
            model,
            peft_config,
            adapter_name="test",
            auto_find_all_linears=False,
            train_mode=True,
        )
        linear_layer = model_lora.base_model.model.model.layers[0].self_attn.qkv_proj
        self.assertTrue(isinstance(linear_layer, GPTQLoraLinear))

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.check_model_trainable(model_lora, tokenizer)

    def test_adalora_conversion(self):
        model = AutoGPTQForCausalLM.from_quantized(
            MODEL_NAME,
            use_triton=False,
            warmup_triton=False,
            trainable=True,
            inject_fused_attention=True,
            inject_fused_mlp=False,
            use_safetensors=True,
        )
        peft_config = GPTQAdaLoraConfig(
            init_r=20,
            target_r=16,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["qkv_proj"],
        )
        model_lora = get_gptq_peft_model(
            model,
            peft_config,
            adapter_name="test",
            auto_find_all_linears=False,
            train_mode=True,
        )
        linear_layer = model_lora.base_model.model.model.layers[0].self_attn.qkv_proj
        self.assertTrue(isinstance(linear_layer, GPTQSVDLinear))

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.check_model_trainable(model_lora, tokenizer)
