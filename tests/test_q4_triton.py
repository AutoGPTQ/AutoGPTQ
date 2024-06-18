import unittest  # noqa: E402

import torch  # noqa: E402
from auto_gptq_next.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as TritonV2QuantLinear  # noqa: E402

try:
    from exllama_kernels import prepare_buffers, set_tuning_params  # noqa: E402
except ImportError as e:
    print(f"[WARNING] Could not load exllama_kernels: {e}")

from auto_gptq_next import AutoGPTQForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestsQ4Triton(unittest.TestCase):
    def test_generation_no_act_order(self):
        prompt = "I am in Paris and"

        reference_output = "<s> I am in Paris and I am going to the Louvre Museum. What time does it open and what is the best way to get there?\nThe Louvre Museum in Paris is open from 9:00 AM to 6:00 PM every day except for Tuesdays. The best way to get"
        new_tokens = 60

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"

        model_q = AutoGPTQForCausalLM.from_quantized(
            model_id,
            device="cuda:0",
            use_triton=True,
            disable_exllama=True,
            disable_exllamav2=True,
            torch_dtype=torch.float16,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

    def test_generation_with_act_order(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and it is a beautiful day. I am sitting in a café, drinking coffee and writing this book. I am surrounded by the sights and sounds of the city, and I am filled with a sense of contentment and gratitude.\n\nI am grateful for the opportunity to live and"

        model_id = "TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g"
        revision = "actorder"
        model_basename = "vicuna-13B-1.1-GPTQ-4bit-128g.latest"

        model_q = AutoGPTQForCausalLM.from_quantized(
            model_id,
            revision=revision,
            device="cuda:0",
            use_triton=True,
            model_basename=model_basename,
            disable_exllama=True,
            disable_exllamav2=True,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)
