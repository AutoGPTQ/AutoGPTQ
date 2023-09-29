from unittest import TestCase
from peft import TaskType
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.utils.peft_utils import GPTQLoraConfig, get_gptq_peft_model, GPTQLoraLinear


class TestPeftConversion(TestCase):
    def testLoraConversion(self):
        model = AutoGPTQForCausalLM.from_quantized(
            "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ",
            use_triton=False,
            warmup_triton=False,
            trainable=True,
            inject_fused_attention=True,
            inject_fused_mlp=False,
            use_safetensors=True
        )
        #model.quantization_method = "gptq"
        #model.config.quantization_config = model.quantize_config
        #model.model.quantization_method = "gptq"
        #model.model.config.quantization_config = model.quantize_config
        #model.quantize_config.disable_exllama = True
        peft_config = GPTQLoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["qkv_proj"],
        )
        model_lora = get_gptq_peft_model(
            model,
            peft_config,
            adapter_name="test",
            auto_find_all_linears=False,
            train_mode=True,
        )
        linear_layer = model_lora.base_model.model.model.layers[0].self_attn.qkv_proj
        assert isinstance(linear_layer, GPTQLoraLinear)
