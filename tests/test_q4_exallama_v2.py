import unittest  # noqa: E402

import torch  # noqa: E402
from auto_gptq_next.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear  # noqa: E402
from auto_gptq_next.utils.import_utils import dynamically_import_QuantLinear  # noqa: E402

from .test_q4_exallama import CUDA_OLD_REFERENCE

try:
    from exllama_kernels import prepare_buffers, set_tuning_params  # noqa: E402
except ImportError as e:
    print(f"[WARNING] Could not load exllama_kernels: {e}")

from auto_gptq_next import AutoGPTQForCausalLM  # noqa: E402
from auto_gptq_next.models._utils import autogptq_post_init  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from .test_q4_cuda import get_diff


class TestsQ4ExllamaV2(unittest.TestCase):
    def test_exllamav2(self):
        group_size = 128

        m = 1
        k = 1024
        n = 1024
        device = torch.device("cuda:0")

        linear_class = dynamically_import_QuantLinear(use_triton=False, desc_act=False, group_size=group_size, bits=4)

        linear = linear_class(
            bits=4,
            group_size=group_size,
            infeatures=k,
            outfeatures=n,
            bias=False,
        )

        self.assertTrue(isinstance(linear, QuantLinear))

        torch.manual_seed(42)

        linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
        linear.scales = linear.scales + 0.002
        linear.qzeros += 0b00010001000100010001000100010001  # for new weight format

        linear = linear.eval()
        linear = linear.to(device)

        linear = autogptq_post_init(linear, use_act_order=False)

        inp = torch.rand(1, m, k, dtype=torch.float16).to(device)

        with torch.no_grad():
            res = linear(inp)[0][0]

        reference = CUDA_OLD_REFERENCE.to(device)

        self.assertTrue(
            torch.allclose(res, reference, rtol=3e-5, atol=2e-2),
            get_diff(res, reference),
        )

    def test_generation_no_act_order(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am going to the Louvre Museum. What time does it open and what is the best way to get there?\nThe Louvre Museum in Paris is open from 9:00 AM to 6:00 PM every day except for Tuesdays. The best way to get"

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"

        model_q = AutoGPTQForCausalLM.from_quantized(model_id, device="cuda:0", use_triton=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

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
            use_triton=False,
            model_basename=model_basename,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_exllama_v2_buffer_size(self):
        # prompt = "I'm in Paris and" * 450
        prompt = "I'm in Paris and" * 500
        device = torch.device("cuda:0")

        model_id = "TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g"
        revision = "actorder"
        model_basename = "vicuna-13B-1.1-GPTQ-4bit-128g.latest"

        model_q = AutoGPTQForCausalLM.from_quantized(
            model_id,
            revision=revision,
            device="cuda:0",
            use_triton=False,
            model_basename=model_basename,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        self.assertTrue(inp["input_ids"].shape[1] > 2048)  # 2048 is the default max_input_length for LLama

        _ = model_q.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
