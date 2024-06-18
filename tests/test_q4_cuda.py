import unittest  # noqa: E402

import torch  # noqa: E402
from auto_gptq_next.utils.import_utils import dynamically_import_QuantLinear  # noqa: E402
from parameterized import parameterized  # noqa: E402

try:
    from exllama_kernels import prepare_buffers, set_tuning_params  # noqa: E402
except ImportError as e:
    print(f"[WARNING] Could not load exllama_kernels: {e}")

from auto_gptq_next import AutoGPTQForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def get_diff(a, ref):
    eps = 1e-6
    return f"Maxdiff: {(a - ref).abs().max()}, Mean relative diff: {((a - ref).abs() / (ref.abs() + eps)).mean()}"


class TestsQ4CUDA(unittest.TestCase):
    REFERENCE_OLD_HALF = torch.Tensor(
        [
            1.5332,
            2.1250,
            1.7910,
            1.8008,
            1.9688,
            1.3262,
            1.7627,
            1.8164,
            1.9307,
            1.8574,
            1.5449,
            1.5293,
            1.6074,
            1.5566,
            1.8545,
            1.6582,
            1.8838,
            2.0215,
            1.8525,
            1.2920,
            1.9561,
            2.2617,
            1.7891,
            2.2656,
            1.6543,
            2.0566,
            1.4756,
            1.1826,
            1.8174,
            2.1191,
            1.6641,
            2.0586,
            1.6182,
            1.7627,
            1.7920,
            1.4424,
            2.0723,
            1.6865,
            1.2979,
            2.0840,
            1.6729,
            1.9648,
            2.1602,
            1.6006,
            1.2773,
            2.2129,
            1.8057,
            1.7285,
            1.6621,
            1.6475,
            1.4805,
            1.7959,
            1.5010,
            0.8643,
            2.6680,
            2.0918,
            1.8555,
            1.9795,
            1.3271,
            1.8359,
            1.6338,
            1.9766,
            1.7881,
            1.6025,
            1.7637,
            1.7012,
            1.7852,
            1.5674,
            0.8091,
            1.7188,
            1.6123,
            1.8525,
            1.4434,
            1.9590,
            1.5801,
            1.4209,
            1.7178,
            1.8408,
            2.4141,
            1.9658,
            1.4922,
            2.1992,
            1.9473,
            1.8047,
            1.2979,
            1.6396,
            1.6221,
            1.5020,
            1.9941,
            1.7725,
            1.6064,
            1.5449,
            1.8418,
            1.2656,
            1.4824,
            1.7734,
            2.0098,
            1.7197,
            1.7686,
            1.4160,
            1.7275,
            2.1738,
            1.9609,
            1.7686,
            1.6396,
            2.1465,
            1.2188,
            1.2002,
            2.1113,
            1.7227,
            1.5811,
            1.7607,
            2.2773,
            1.8945,
            1.4111,
            1.5801,
            1.7744,
            2.0684,
            2.1621,
            1.8027,
            1.1045,
            1.9648,
            2.2402,
            2.0742,
            1.3330,
            1.5840,
            2.1465,
            2.0176,
            1.5068,
            1.9834,
            1.7725,
            1.5527,
            1.7803,
            1.7744,
            1.5312,
            1.2695,
            1.9209,
            2.0469,
            1.6777,
            2.5215,
            1.8389,
            1.7598,
            1.5498,
            1.6807,
            1.7324,
            1.5938,
            1.9268,
            1.7734,
            1.4463,
            2.0391,
            2.0527,
            2.2129,
            1.6787,
            2.0586,
            1.8975,
            1.5713,
            1.6992,
            1.8770,
            1.7207,
            1.7080,
            1.1611,
            1.8584,
            2.4570,
            1.6016,
            1.4834,
            1.1777,
            1.7969,
            1.8955,
            1.8906,
            1.6738,
            1.7510,
            1.4316,
            1.8340,
            2.2461,
            1.7744,
            2.1934,
            1.4824,
            1.8828,
            1.6387,
            2.4629,
            1.8887,
            1.5137,
            1.4648,
            1.6406,
            1.7188,
            2.2656,
            1.5801,
            2.1484,
            2.0625,
            2.0098,
            1.7549,
            1.1768,
            1.4385,
            2.0723,
            1.6172,
            1.7832,
            1.8301,
            1.6064,
            1.5215,
            1.9297,
            2.3750,
            2.1504,
            1.7070,
            1.1289,
            1.4473,
            1.5674,
            1.6836,
            2.2930,
            1.1221,
            1.5557,
            1.7559,
            1.8281,
            2.0703,
            1.9443,
            2.0684,
            2.2988,
            1.6348,
            2.3379,
            2.4414,
            1.8857,
            2.0039,
            1.4844,
            1.5488,
            1.6514,
            2.3711,
            1.9941,
            2.3066,
            1.4287,
            2.1777,
            1.6445,
            1.6025,
            1.5938,
            1.5508,
            1.9502,
            2.1309,
            1.2666,
            1.1523,
            1.9561,
            1.8584,
            1.9746,
            1.5986,
            1.9688,
            2.1973,
            1.1523,
            2.3281,
            1.2451,
            1.8447,
            2.2051,
            1.5254,
            1.5342,
            2.1016,
            1.6523,
            1.6279,
            1.1680,
            1.3037,
            2.1035,
        ]
    ).to(torch.float16)

    REFERENCE_OLD_NO_HALF = torch.Tensor(
        [
            1.5332,
            2.1250,
            1.7910,
            1.7998,
            1.9678,
            1.3262,
            1.7617,
            1.8154,
            1.9307,
            1.8574,
            1.5449,
            1.5293,
            1.6074,
            1.5557,
            1.8545,
            1.6582,
            1.8838,
            2.0195,
            1.8525,
            1.2920,
            1.9561,
            2.2617,
            1.7891,
            2.2656,
            1.6543,
            2.0566,
            1.4756,
            1.1826,
            1.8164,
            2.1191,
            1.6641,
            2.0586,
            1.6182,
            1.7617,
            1.7920,
            1.4424,
            2.0723,
            1.6865,
            1.2969,
            2.0840,
            1.6729,
            1.9639,
            2.1602,
            1.5996,
            1.2773,
            2.2129,
            1.8057,
            1.7275,
            1.6621,
            1.6475,
            1.4805,
            1.7949,
            1.5010,
            0.8643,
            2.6680,
            2.0918,
            1.8545,
            1.9795,
            1.3271,
            1.8350,
            1.6338,
            1.9766,
            1.7881,
            1.6025,
            1.7637,
            1.7012,
            1.7842,
            1.5664,
            0.8086,
            1.7188,
            1.6113,
            1.8516,
            1.4434,
            1.9590,
            1.5801,
            1.4209,
            1.7168,
            1.8408,
            2.4141,
            1.9658,
            1.4922,
            2.1973,
            1.9463,
            1.8047,
            1.2979,
            1.6396,
            1.6221,
            1.5010,
            1.9941,
            1.7725,
            1.6064,
            1.5449,
            1.8418,
            1.2656,
            1.4824,
            1.7734,
            2.0098,
            1.7188,
            1.7686,
            1.4160,
            1.7266,
            2.1738,
            1.9600,
            1.7686,
            1.6396,
            2.1465,
            1.2188,
            1.2002,
            2.1113,
            1.7227,
            1.5811,
            1.7598,
            2.2773,
            1.8936,
            1.4102,
            1.5801,
            1.7734,
            2.0684,
            2.1621,
            1.8027,
            1.1045,
            1.9648,
            2.2402,
            2.0742,
            1.3330,
            1.5840,
            2.1465,
            2.0176,
            1.5068,
            1.9834,
            1.7725,
            1.5527,
            1.7793,
            1.7744,
            1.5312,
            1.2695,
            1.9209,
            2.0469,
            1.6777,
            2.5195,
            1.8389,
            1.7598,
            1.5498,
            1.6797,
            1.7324,
            1.5928,
            1.9258,
            1.7734,
            1.4463,
            2.0391,
            2.0508,
            2.2129,
            1.6787,
            2.0586,
            1.8975,
            1.5713,
            1.6992,
            1.8770,
            1.7207,
            1.7070,
            1.1602,
            1.8584,
            2.4570,
            1.6016,
            1.4834,
            1.1777,
            1.7959,
            1.8955,
            1.8906,
            1.6738,
            1.7510,
            1.4316,
            1.8330,
            2.2461,
            1.7744,
            2.1934,
            1.4824,
            1.8828,
            1.6387,
            2.4629,
            1.8887,
            1.5137,
            1.4648,
            1.6406,
            1.7178,
            2.2637,
            1.5801,
            2.1484,
            2.0605,
            2.0098,
            1.7539,
            1.1768,
            1.4375,
            2.0723,
            1.6162,
            1.7832,
            1.8291,
            1.6064,
            1.5215,
            1.9297,
            2.3750,
            2.1504,
            1.7061,
            1.1289,
            1.4473,
            1.5674,
            1.6836,
            2.2930,
            1.1221,
            1.5547,
            1.7559,
            1.8281,
            2.0703,
            1.9443,
            2.0684,
            2.2988,
            1.6348,
            2.3379,
            2.4414,
            1.8857,
            2.0020,
            1.4834,
            1.5488,
            1.6514,
            2.3711,
            1.9941,
            2.3047,
            1.4277,
            2.1777,
            1.6445,
            1.6025,
            1.5938,
            1.5508,
            1.9502,
            2.1309,
            1.2666,
            1.1514,
            1.9551,
            1.8584,
            1.9746,
            1.5986,
            1.9688,
            2.1953,
            1.1514,
            2.3262,
            1.2451,
            1.8447,
            2.2051,
            1.5254,
            1.5342,
            2.1016,
            1.6523,
            1.6279,
            1.1680,
            1.3037,
            2.1035,
        ]
    ).to(torch.float16)

    @parameterized.expand([(False,), (True,)])
    def test_cuda_old(self, use_half2: bool):
        group_size = 128

        # test the 256 kernel (in_features % 256 == 0 and out_features % 256 == 0)
        m = 1
        k = 256
        n = 256
        device = "cuda"

        linear_class = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=False,
            group_size=group_size,
            bits=4,
            disable_exllama=True,
            disable_exllamav2=True,
        )

        weight_dtype = torch.float16 if use_half2 else torch.float32
        linear = linear_class(
            bits=4,
            group_size=group_size,
            infeatures=k,
            outfeatures=n,
            bias=False,
            weight_dtype=weight_dtype,
        )

        torch.manual_seed(42)

        linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
        linear.scales = linear.scales + 0.002
        linear.qzeros += 0b00010001000100010001000100010001  # for new weight format
        linear.use_cuda_fp16 = use_half2
        self.assertTrue(linear.autogptq_cuda_available)

        # We cast twice just for the seed.
        inp = torch.rand(1, m, k, dtype=torch.float16).to(device).to(weight_dtype)

        linear = linear.eval()
        linear = linear.to(device)

        with torch.no_grad():
            res = linear(inp)[0][0]

        if use_half2:
            reference = self.REFERENCE_OLD_HALF.to(device).to(weight_dtype)
        else:
            reference = self.REFERENCE_OLD_NO_HALF.to(device).to(weight_dtype)

        self.assertTrue(torch.allclose(res, reference, rtol=1e-3), get_diff(res, reference))

    @parameterized.expand(
        [
            (torch.float32, "cpu"),
            (torch.float32, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_with_act_order(self, torch_dtype, device):
        prompt = "I am in Paris and"

        # Reference generated with the cuda-old kernel
        if device == "cpu":
            # CPU implementation is extremely slow.
            new_tokens = 2
            reference_output = "<s> I am in Paris and it is"
        else:
            reference_output = "<s> I am in Paris and it is a beautiful day. I am sitting in a café, drinking coffee and writing this book. I am surrounded by the sights and sounds of the city, and I am filled with a sense of contentment and gratitude.\n\nI am grateful for the opportunity to live and"
            new_tokens = 60

        model_id = "TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g"
        revision = "actorder"
        model_basename = "vicuna-13B-1.1-GPTQ-4bit-128g.latest"

        model_q = AutoGPTQForCausalLM.from_quantized(
            model_id,
            revision=revision,
            device=device,
            use_triton=False,
            model_basename=model_basename,
            disable_exllama=True,
            disable_exllamav2=True,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

    @parameterized.expand(
        [
            (torch.float32, "cpu"),
            (torch.float32, "cuda:0"),
            (torch.float16, "cuda:0"),
        ]
    )
    def test_generation_no_act_order(self, torch_dtype, device):
        prompt = "I am in Paris and"

        # Reference generated with the cuda-old kernel
        if device == "cpu":
            # CPU implementation is extremely slow.
            new_tokens = 3
            reference_output = "<s> I am in Paris and I am going"
        else:
            reference_output = "<s> I am in Paris and I am going to the Louvre Museum. What time does it open and what is the best way to get there?\nThe Louvre Museum in Paris is open from 9:00 AM to 6:00 PM every day except for Tuesdays. The best way to get"
            new_tokens = 60

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"

        model_q = AutoGPTQForCausalLM.from_quantized(
            model_id,
            device=device,
            use_triton=False,
            disable_exllama=True,
            disable_exllamav2=True,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)
