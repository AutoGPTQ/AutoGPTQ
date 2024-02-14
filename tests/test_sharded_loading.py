import unittest

import torch
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM


class TestShardedLoading(unittest.TestCase):

    def test_loading(self):
        model_name = "TheBlokeAI/llama-68m-GPTQ-sharded"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            use_triton=False,
            device='cuda:0',
            warmup_triton=False,
            disable_exllama=True,
            disable_exllamav2=True,
            inject_fused_attention=True,
            inject_fused_mlp=False,
            use_safetensors=True
        )

        tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
        result = tokenizer.decode(tokens)

        test_tokens = torch.tensor([    1, 29871, 29896, 29941, 29941, 29955, 29955, 29955, 29955, 29955,
        29955, 29955, 29955, 29955, 29955, 29955, 29955, 29955, 29955, 29955,
        29955, 29955, 29955, 29955, 29955, 29955], device='cuda:0')

        assert(result == '<s> 133777777777777777777777')
        assert(torch.equal(tokens, test_tokens))
