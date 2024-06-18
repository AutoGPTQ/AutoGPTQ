import json
import os
import tempfile
import unittest

from auto_gptq_next import AutoGPTQNext
from auto_gptq_next.quantization import FORMAT, FORMAT_FIELD, QUANT_CONFIG_FILENAME


class TestSerialization(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_marlin_local_serialization(self):
        model = AutoGPTQNext.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "gptq_model-4bit-128g.safetensors")))

            with open(os.path.join(tmpdir, QUANT_CONFIG_FILENAME), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(config[FORMAT_FIELD] == FORMAT.MARLIN)

            model = AutoGPTQNext.from_quantized(tmpdir, device="cuda:0", use_marlin=True)

    def test_marlin_hf_cache_serialization(self):
        model = AutoGPTQNext.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

        model = AutoGPTQNext.from_quantized(self.MODEL_ID, device="cuda:0", use_marlin=True)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

    def test_gptq_v1_to_v2_runtime_convert(self):
        model = AutoGPTQNext.from_quantized(self.MODEL_ID, device="cuda:0")
        self.assertTrue(model.quantize_config.format == FORMAT.GPTQ_V2)

    def test_gptq_v1_serialization(self):
        model = AutoGPTQNext.from_quantized(self.MODEL_ID, device="cuda:0")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir, format="gptq")

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as f:
                quantize_config = json.load(f)

            self.assertTrue(quantize_config["format"] == "gptq")
