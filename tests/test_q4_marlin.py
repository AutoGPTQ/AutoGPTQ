import unittest  # noqa: E402

import torch  # noqa: E402

from auto_gptq_next.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear  # noqa: E402


try:
    from exllama_kernels import prepare_buffers, set_tuning_params  # noqa: E402
except ImportError as e:
    print(f"[WARNING] Could not load exllama_kernels: {e}")

from transformers import AutoTokenizer  # noqa: E402

from auto_gptq_next import AutoGPTQForCausalLM  # noqa: E402


class TestQ4Marlin(unittest.TestCase):
    def test_generation(self):
        # Reference generated with the cuda-old kernel and TheBloke/Llama-2-7B-Chat-GPTQ
        reference_output = "<s> I am in Paris and I am feeling very sad and lonely. everybody I know is busy and I don't have any friends here. I am staying in a small apartment in the 11th arrondissement and I am feeling very isolated. I miss my friends and family back home and I don'"

        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"

        try:
            model_q = AutoGPTQForCausalLM.from_quantized(model_id, device="cuda:0", use_marlin=True)
        except ValueError as e:
            if torch.version.hip:
                self.assertTrue("Can not use Marlin int4*fp16 kernel with AMD ROCm" in e.text)
                self.skipTest("Can not run this test on ROCm")
            else:
                raise e

        has_marlin = False
        for _, module in model_q.named_modules():
            if isinstance(module, MarlinQuantLinear):
                has_marlin = True
                break
        self.assertTrue(has_marlin)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "s3nh/starcoderbase-1b-GPTQ"
        try:
            model_q = AutoGPTQForCausalLM.from_quantized(model_id, device="cuda:0", use_marlin=True)
        except ValueError as e:
            if torch.version.hip:
                self.assertTrue("Can not use Marlin int4*fp16 kernel with AMD ROCm" in e.text)
                self.skipTest("Can not run this test on ROCm")
            else:
                raise e

        for _, param in model_q.named_parameters():
            self.assertTrue(param.device != torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertTrue(param.device != torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        tokenizer = AutoTokenizer.from_pretrained("Xenova/starcoderbase-1b")

        prompt = "Today I am in Paris and"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertTrue(predicted_text.startswith("Today I am in Paris and I am a student of the Master's"))
