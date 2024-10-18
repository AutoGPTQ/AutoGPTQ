import json
import os
import tempfile
import time
import unittest

from parameterized import parameterized
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.quantization import CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_FIELD, QUANT_CONFIG_FILENAME
from auto_gptq.quantization.config import QUANT_METHOD, BaseQuantizeConfig


class TestSerialization(unittest.TestCase):
    MODEL_ID = "habanoz/TinyLlama-1.1B-Chat-v0.3-GPTQ"

    def setUp(self):
        dummy_config = BaseQuantizeConfig(
            model_name_or_path=self.MODEL_ID,
            quant_method=QUANT_METHOD.GPTQ,
            checkpoint_format=CHECKPOINT_FORMAT.MARLIN)

        model_cache_path, is_cached = dummy_config.get_cache_file_path()

        if is_cached:
            os.remove(model_cache_path)

    @parameterized.expand([("cuda:0"), ("cpu")])
    def test_marlin_and_ipex_local_serialization(self, device: str):
        checkpoint_format = CHECKPOINT_FORMAT.MARLIN if device == "cuda:0" else CHECKPOINT_FORMAT.GPTQ
        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device=device,
                                                   use_marlin=True if device == "cuda:0" else False,
                                                   use_ipex=True if device == "cpu" else False)
        end = time.time()
        first_load_time = end - start

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.safetensors")))
            model_cache_path, is_cached = model.quantize_config.get_cache_file_path()
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, model_cache_path)))

            with open(os.path.join(tmpdir, QUANT_CONFIG_FILENAME), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(model.quantize_config.checkpoint_format == checkpoint_format)

            start = time.time()
            model = AutoGPTQForCausalLM.from_quantized(tmpdir, device=device,
                                                       use_marlin=True if device == "cuda:0" else False,
                                                       use_ipex=True if device == "cpu" else False)
            end = time.time()
            second_load_time = end - start

        # Since we use a CUDA kernel to repack weights, the first load time is already small.
        self.assertTrue(second_load_time < first_load_time)

    @parameterized.expand([("cuda:0"), ("cpu")])
    def test_marlin_and_ipex_hf_cache_serialization(self, device: str):
        checkpoint_format = CHECKPOINT_FORMAT.MARLIN if device == "cuda:0" else CHECKPOINT_FORMAT.GPTQ
        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device=device,
                                                   use_marlin=True if device == "cuda:0" else False,
                                                   use_ipex=True if device == "cpu" else False)
        self.assertTrue(model.quantize_config.checkpoint_format == checkpoint_format)
        end = time.time()
        first_load_time = end - start

        model_cache_path, is_cached = model.quantize_config.get_cache_file_path()
        self.assertTrue("assets" in model_cache_path)
        self.assertTrue(is_cached)

        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device=device,
                                                   use_marlin=True if device == "cuda:0" else False,
                                                   use_ipex=True if device == "cpu" else False)
        self.assertTrue(model.quantize_config.checkpoint_format == checkpoint_format)
        end = time.time()
        second_load_time = end - start

        # Since we use a CUDA kernel to repack weights, the first load time is already small.
        self.assertTrue(second_load_time < first_load_time)
