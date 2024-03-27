import json
import os
import tempfile
import time
import unittest

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

    def test_marlin_local_serialization(self):
        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        end = time.time()
        first_load_time = end - start

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.safetensors")))
            model_cache_path, is_cached = model.quantize_config.get_cache_file_path()
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, model_cache_path)))

            with open(os.path.join(tmpdir, QUANT_CONFIG_FILENAME), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(config[CHECKPOINT_FORMAT_FIELD] == CHECKPOINT_FORMAT.MARLIN)

            start = time.time()
            model = AutoGPTQForCausalLM.from_quantized(tmpdir, device="cuda:0", use_marlin=True)
            end = time.time()
            second_load_time = end - start

        # Since we use a CUDA kernel to repack weights, the first load time is already small.
        self.assertTrue(second_load_time < first_load_time)

    def test_marlin_hf_cache_serialization(self):
        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.checkpoint_format == CHECKPOINT_FORMAT.MARLIN)
        end = time.time()
        first_load_time = end - start

        model_cache_path, is_cached = model.quantize_config.get_cache_file_path()
        self.assertTrue("assets" in model_cache_path)
        self.assertTrue(is_cached)

        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.checkpoint_format == CHECKPOINT_FORMAT.MARLIN)
        end = time.time()
        second_load_time = end - start

        # Since we use a CUDA kernel to repack weights, the first load time is already small.
        self.assertTrue(second_load_time < first_load_time)
