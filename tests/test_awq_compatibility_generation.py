# ruff: noqa: I001
import unittest

import torch
import autogptq_cuda_64
import autogptq_cuda_256
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQLinear


try:
    from awq import AutoAWQForCausalLM
except ModuleNotFoundError as e:
    AutoAWQForCausalLM = None
    AWQ_EXCEPTION = e


class TestAwqCompatibility(unittest.TestCase):
    # TODO: test cuda-old fp16.
    # TODO: test cuda-old fp32.
    # TODO: test exllama v2.

    def test_generation_cuda_old_fp32_pytorch(self):
        if AutoAWQForCausalLM is None:
            self.skipTest(
                f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this test. {AWQ_EXCEPTION}"
            )

        device = torch.device("cuda:0")
        quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        model_autogptq = AutoGPTQForCausalLM.from_quantized(
            quant_path,
            device=device,
            use_triton=False,
            disable_exllama=True,
            disable_exllamav2=True,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(quant_path)

        prompt = "I am in Paris and I am going to see the"

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        for name, submodule in model_autogptq.named_modules():
            if isinstance(submodule, CudaOldQLinear):
                # Just a hack to test the handmade pytorch implementation path.
                submodule.autogptq_cuda_available = False

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

    def test_generation_cuda_old_cuda_256(self):
        if AutoAWQForCausalLM is None:
            self.skipTest(
                f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this test. {AWQ_EXCEPTION}"
            )

        device = torch.device("cuda:0")
        quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        prompt = "I am in Paris and I am going to see the"

        for torch_dtype in [torch.float16, torch.float32]:
            model_autogptq = AutoGPTQForCausalLM.from_quantized(
                quant_path,
                device=device,
                use_triton=False,
                disable_exllama=True,
                disable_exllamav2=True,
                torch_dtype=torch_dtype,
            )

            for name, module in model_autogptq.named_modules():
                if isinstance(module, CudaOldQLinear):
                    self.assertTrue(module.autogptq_cuda == autogptq_cuda_256)

                    if torch_dtype == torch.float32:
                        self.assertFalse(module.use_cuda_fp16)
                    else:
                        self.assertTrue(module.use_cuda_fp16)

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

    def test_generation_cuda_old_cuda_64(self):
        if AutoAWQForCausalLM is None:
            self.skipTest(
                f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this test. {AWQ_EXCEPTION}"
            )

        device = torch.device("cuda:0")
        quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        prompt = "I am in Paris and I am going to see the"

        for torch_dtype in [torch.float16, torch.float32]:
            model_autogptq = AutoGPTQForCausalLM.from_quantized(
                quant_path,
                device=device,
                use_triton=False,
                disable_exllama=True,
                disable_exllamav2=True,
                torch_dtype=torch_dtype,
            )

            # Force autogptq_cuda_64.
            for name, module in model_autogptq.named_modules():
                if isinstance(module, CudaOldQLinear):
                    if module.autogptq_cuda != autogptq_cuda_64:
                        module.autogptq_cuda = autogptq_cuda_64

                    if torch_dtype == torch.float32:
                        self.assertFalse(module.use_cuda_fp16)
                    else:
                        self.assertTrue(module.use_cuda_fp16)

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

    def test_generation_exllama(self):
        if AutoAWQForCausalLM is None:
            self.skipTest(
                f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this test. {AWQ_EXCEPTION}"
            )

        device = torch.device("cuda:0")
        quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        model_autogptq = AutoGPTQForCausalLM.from_quantized(
            quant_path,
            device=device,
            use_triton=False,
            disable_exllama=False,
            disable_exllamav2=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(quant_path)

        prompt = "I am in Paris and I am going to see the"

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        for name, submodule in model_autogptq.named_modules():
            if isinstance(submodule, CudaOldQLinear):
                # Just a hack to test the handmade pytorch implementation path.
                submodule.autogptq_cuda_available = False

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
        print(f"AWQ output: {awq_output}")
        print(f"AutoGPTQ output: {autogptq_output}")

        self.assertTrue(awq_output == autogptq_output)

    def test_generation_exllamav2(self):
        if AutoAWQForCausalLM is None:
            self.skipTest(
                f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this test. {AWQ_EXCEPTION}"
            )

        device = torch.device("cuda:0")
        quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

        model_autogptq = AutoGPTQForCausalLM.from_quantized(
            quant_path,
            device=device,
            use_triton=False,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(quant_path)

        prompt = "I am in Paris and I am going to see the"

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        for name, submodule in model_autogptq.named_modules():
            if isinstance(submodule, CudaOldQLinear):
                # Just a hack to test the handmade pytorch implementation path.
                submodule.autogptq_cuda_available = False

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
        print(f"AWQ output: {awq_output}")
        print(f"AutoGPTQ output: {autogptq_output}")

        self.assertTrue(awq_output == autogptq_output)
