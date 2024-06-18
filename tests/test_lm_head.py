import random
import unittest

import numpy
import torch
from auto_gptq_next import AutoGPTQForCausalLM
from transformers import AutoTokenizer


class TestLmHead(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"

    def setup(self):
        seed = 898
        # stabilize generation
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_load(self):
        prompt = "My name is Lewis and I like to"

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        inputs = tokenizer(prompt, return_tensors="pt").to(device=self.DEVICE)

        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, use_safetensors=True, device=self.DEVICE)

        # validate lm_head is loaded as quantized layer
        assert model.lm_head.__class__.__name__ == "QuantLinear"

        res = model.model.generate(
            **inputs, num_beams=1, min_new_tokens=1, max_new_tokens=128, repetition_penalty=1.25
        )
        res_str = tokenizer.decode(res[0])

        print(f"prompt: {prompt}")
        print(f"result: {res_str}")

        # validated on 4090 and a100 + cuda 12.4 + torch 2.2.2 + transformers 4.40.1
        assert "My name is Lewis and I like to play football." in res_str
