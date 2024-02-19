import unittest

from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM


class TestShardedLoading(unittest.TestCase):

    def test_loading(self):
        model_name = "TheBlokeAI/llama-68m-GPTQ-sharded"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(model_name, device='cuda:0',)

        tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
        result = tokenizer.decode(tokens)

        self.assertTrue(result == '<s> 133777777777777777777777')

    def test_loading_large(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int4")

        model = AutoGPTQForCausalLM.from_quantized("Qwen/Qwen1.5-7B-Chat-GPTQ-Int4", device='cuda:0')

        tokens = model.generate(**tokenizer("Today I am in Paris and", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
        result = tokenizer.decode(tokens)

        self.assertTrue(result == 'Today I am in Paris and I am going to the Louvre Museum. I want to see the Mona Lisa painting, but I')
