import json
import os
import tempfile
import unittest

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.quantization import FORMAT, FORMAT_FIELD, QUANT_CONFIG_FILENAME
from auto_gptq.quantization.config import QUANT_METHOD, BaseQuantizeConfig


class TestSerialization(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def setUp(self):
        dummy_config = BaseQuantizeConfig(
            model_name_or_path=self.MODEL_ID,
            quant_method=QUANT_METHOD.GPTQ,
            format=FORMAT.MARLIN,
        )

        model_cache_path, is_cached = dummy_config.get_cache_file_path()

        if is_cached:
            os.remove(model_cache_path)

    def test_marlin_local_serialization(self):
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "gptq_model-4bit-128g.safetensors")))

            with open(os.path.join(tmpdir, QUANT_CONFIG_FILENAME), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(config[FORMAT_FIELD] == FORMAT.MARLIN)

            model = AutoGPTQForCausalLM.from_quantized(tmpdir, device="cuda:0", use_marlin=True)

    def test_marlin_hf_cache_serialization(self):
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

        model_cache_path, is_cached = model.quantize_config.get_cache_file_path()
        self.assertTrue("assets" in model_cache_path)
        self.assertTrue(is_cached)

        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

    def test_gptq_v1_to_v2_runtime_convert(self):
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0")
        self.assertTrue(model.quantize_config.format == FORMAT.GPTQ_V2)

    def test_gptq_v1_serialization(self):
        model = AutoGPTQForCausalLM.from_quantized(self.MODEL_ID, device="cuda:0")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir, format="gptq")

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as f:
                quantize_config = json.load(f)

            self.assertTrue(quantize_config["format"] == "gptq")
