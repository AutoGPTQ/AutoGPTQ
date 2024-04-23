import os
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch.cuda  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from auto_gptq import AutoGPTQForCausalLM  # noqa: E402
from auto_gptq.quantization import CHECKPOINT_FORMAT, QUANT_CONFIG_FILENAME, BaseQuantizeConfig  # noqa: E402


class TestQuantization(unittest.TestCase):
    @parameterized.expand([
        (False, True, CHECKPOINT_FORMAT.GPTQ_V2),
        (False, False, CHECKPOINT_FORMAT.GPTQ),
        (True, True, CHECKPOINT_FORMAT.MARLIN),
    ])
    def test_quantize(self, use_marlin: bool, sym: bool, checkpoint_format: CHECKPOINT_FORMAT):
        pretrained_model_dir = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            ),
            tokenizer("Today I am in Paris and it is a wonderful day."),
        ]

        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=sym,
            checkpoint_format=checkpoint_format,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_dir,
            quantize_config=quantize_config,
            use_flash_attention_2=False,
        )

        model.quantize(examples)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(
                tmpdirname,
                use_unsafe_math=True if not sym and checkpoint_format == CHECKPOINT_FORMAT.GPTQ else False,
            )

            model = AutoGPTQForCausalLM.from_quantized(
                tmpdirname,
                device="cuda:0",
                use_marlin=use_marlin,
                use_unsafe_math=True if not sym and checkpoint_format == CHECKPOINT_FORMAT.GPTQ else False,
            )
            del model
            torch.cuda.empty_cache()

            # test compat: 1) with simple dict type 2) is_marlin_format
            compat_quantize_config = {
                "bits": 4,
                "group_size": 128,
                "sym": sym,
                "desc_act": True,
                "is_marlin_format": use_marlin,
            }
            model = AutoGPTQForCausalLM.from_quantized(
                tmpdirname,
                device="cuda:0",
                quantize_config=compat_quantize_config,
                use_unsafe_math=True if not sym and checkpoint_format == CHECKPOINT_FORMAT.GPTQ  else False,
            )
            assert isinstance(model.quantize_config, BaseQuantizeConfig)

            del model
            torch.cuda.empty_cache()

            # test checkpoint_format hint to from_quantized()
            os.remove(f"{tmpdirname}/{QUANT_CONFIG_FILENAME}")

            compat_quantize_config = {
                "bits": 4,
                "group_size": 128,
                "sym": sym,
                "desc_act": True,
            }
            model = AutoGPTQForCausalLM.from_quantized(
                tmpdirname,
                device="cuda:0",
                quantize_config=compat_quantize_config,
                checkpoint_format=checkpoint_format,
                use_unsafe_math=True if not sym and checkpoint_format == CHECKPOINT_FORMAT.GPTQ else False,
            )
            assert isinstance(model.quantize_config, BaseQuantizeConfig)
