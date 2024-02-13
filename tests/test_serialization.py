import json
import os
import tempfile
import time
import unittest

import huggingface_hub

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.utils.marlin_utils import _get_cached_marlin_save_name


class TestSerialization(unittest.TestCase):
    MODEL_ID = "habanoz/TinyLlama-1.1B-Chat-v0.3-GPTQ"

    def setUp(self):
        namespace, subfolder = self.MODEL_ID.split("/")
        assets_path = huggingface_hub.cached_assets_path(
            library_name="auto_gptq", namespace=namespace, subfolder=subfolder
        )
        cached_model_path = os.path.join(assets_path, "autogptq_model.safetensors")

        if os.path.isfile(cached_model_path):
            os.remove(cached_model_path)

    def test_marlin_local_serialization(self):
        start = time.time()
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        end = time.time()

        first_load_time = end - start

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.safetensors")))
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, "autogptq_model.safetensors")))

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(config["is_marlin_format"])

            start = time.time()
            model = AutoGPTQForCausalLM.from_quantized(tmpdir, device="cuda:0", use_marlin=True)
            end = time.time()

            second_load_time = end - start

        self.assertTrue(second_load_time < 0.2 * first_load_time)

    def test_marlin_hf_cache_serialization(self):
        start = time.time()
        _ = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        end = time.time()

        first_load_time = end - start

        model_cache_path = _get_cached_marlin_save_name(self.MODEL_ID)
        self.assertTrue("assets" in model_cache_path)
        self.assertTrue(os.path.isfile(model_cache_path))

        start = time.time()
        _ = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        end = time.time()

        second_load_time = end - start

        self.assertTrue(second_load_time < 0.2 * first_load_time)
