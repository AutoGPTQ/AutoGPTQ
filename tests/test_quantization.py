import tempfile
import unittest

from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


class TestQuantization(unittest.TestCase):
    def test_quantize(self):
        pretrained_model_dir = "saibo/llama-1B"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_dir,
            quantize_config=quantize_config,
            use_flash_attention_2=False,
        )

        model.quantize(examples)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
