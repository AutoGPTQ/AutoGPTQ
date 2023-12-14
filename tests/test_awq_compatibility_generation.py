import torch
from awq import AutoAWQForCausalLM
import unittest
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class TestAwqCompatibility(unittest.TestCase):
    def test_generation(self):
        # TODO: somehow download TheBloke/Llama-2-7B-Chat-AWQ instead of hardcoding path.
        # TODO: test exllama.
        # TODO: test cuda-old fp16.
        # TODO: test cuda-old fp32.
        # TODO: test exllama v2.
        device = torch.device("cuda:0")

        quant_path = "/fsx/felix/llama_7b_awq_gemm"

        model_autogptq = AutoGPTQForCausalLM.from_quantized(quant_path, device=device, use_triton=False, inject_fused_attention=False, inject_fused_mlp=False, disable_exllama=True, disable_exllamav2=True, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)

        prompt = "I am in Paris and I am going to see the"

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        autogptq_output = model_autogptq.model.generate(**inp, num_beams=1, min_new_tokens=30, max_new_tokens=30)
        autogptq_output = tokenizer.decode(autogptq_output[0])

        model_awq = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)

        awq_output = model_awq.generate(
            **inp,
            num_beams=1,
            min_new_tokens=30,
            max_new_tokens=30,
        )

        awq_output = tokenizer.decode(awq_output[0])

        self.assertTrue(awq_output == autogptq_output)