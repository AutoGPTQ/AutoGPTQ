import json
import logging
import os
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch.cuda  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from auto_gptq import AutoGPTQForCausalLM, __version__  # noqa: E402
from auto_gptq.quantization import CHECKPOINT_FORMAT, QUANT_CONFIG_FILENAME, BaseQuantizeConfig  # noqa: E402
from auto_gptq.quantization.config import META_FIELD_QUANTIZER, META_QUANTIZER_AUTOGPTQ

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

        with (tempfile.TemporaryDirectory() as tmpdirname):
            model.save_pretrained(
                tmpdirname,
            )

            logging.info(f"Saved config mem: {model.quantize_config}")

            with open(tmpdirname + "/" + QUANT_CONFIG_FILENAME, 'r') as f:
                file_dict = json.loads(f.read())
                # skip comparison of these two model path specific fields that do not exist in memory
                file_dict["model_name_or_path"] = None
                file_dict["model_file_base_name"] = None

                # make sure the json dict saved to file matches config in memory
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            model = AutoGPTQForCausalLM.from_quantized(
                tmpdirname,
                device="cuda:0",
                use_marlin=use_marlin,
            )

            logging.info(f"Loaded config: {model.quantize_config}")
            assert model.quantize_config.meta_get_versionable(META_FIELD_QUANTIZER) == (META_QUANTIZER_AUTOGPTQ, __version__)
            del model
            torch.cuda.empty_cache()

            # skip compat test with sym=False and v1 since we do meta version safety check
            if not sym and checkpoint_format == CHECKPOINT_FORMAT.GPTQ:
                return

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
            )
            assert isinstance(model.quantize_config, BaseQuantizeConfig)
