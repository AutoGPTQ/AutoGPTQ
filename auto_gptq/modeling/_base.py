import json
from dataclasses import dataclass, field, fields
from logging import getLogger
from os.path import join
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file as safe_load, save_file as safe_save
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from ._const import *
from ._utils import *
from ..quantization import *

logger = getLogger(__name__)


@dataclass
class BaseQuantizeConfig:
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    group_size: int = field(default=-1)

    def __post_init__(self):
        fields_info = fields(self)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")

    def save_pretrained(self, save_dir: str):
        with open(join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        with open(join(save_dir, "quantize_config.json"), "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_dict(self):
        return {
            "bits": self.bits,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "group_size": self.group_size
        }


class BaseGPTQForCausalLM:
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None

    def __init__(self, model: PreTrainedModel, quantized: bool, quantize_config: BaseQuantizeConfig):
        self.model = model
        self.model_type = self.model.config.model_type
        self._quantized = quantized
        self.quantize_config = quantize_config

    @property
    def quantized(self):
        return self._quantized

    def _move_outside_layer_modules(self, device):
        for module_name in self.outside_layer_modules:
            module = get_module_by_name(self.model, module_name)
            if module is not None:
                module.to(device)

    @staticmethod
    def _resize_attention_mask(attention_mask: List[torch.LongTensor]):
        return attention_mask

    def quantize(self, examples: List[Dict[str, torch.LongTensor]]):
        if self.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")

        layer_inputs = []
        attention_masks = []
        layer_outputs = []

        class LayerHijacker(nn.Module):
            """hijack layer's forward pass to cache data"""

            def __init__(self, m):
                super().__init__()
                self.module = m

            def forward(self, inp, **kwargs):
                bsz = inp.size(0)
                for i in range(bsz):
                    layer_inputs.append(inp[i].to(CPU))
                    attention_masks.append(kwargs["attention_mask"][i].to(CPU))
                raise ValueError

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_examples = len(examples)
        layers = get_module_by_name(self.model, self.layers_block_name)

        layers[0] = layers[0].to(CUDA)
        self._move_outside_layer_modules(CUDA)

        # get inputs for first layer
        layers[0] = LayerHijacker(layers[0])
        for example in examples:
            for k, v in example.items():
                if k == "input_ids" and len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = v.to(CUDA)
            try:
                self.model(**example)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        self._move_outside_layer_modules(CPU)

        torch.cuda.empty_cache()

        # resize attention mask for some special models
        attention_masks = self._resize_attention_mask(attention_masks)

        quantizers = {}
        for i in range(len(layers)):
            logger.info(f"Start quantizing layer {i + 1}/{len(layers)}")
            layer = layers[i].to(CUDA)

            full = find_layers(layer)
            for names in self.inside_layer_modules:
                subset = {n: full[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        self.quantize_config.bits,
                        perchannel=True,
                        sym=True,
                        mse=False
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_examples):
                    layer_input = layer_inputs[j].unsqueeze(0).to("cuda:0")
                    layer_attention_mask = attention_masks[j].to("cuda:0")
                    layer(layer_input, attention_mask=layer_attention_mask)
                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f'Quantizing {name} in layer {i + 1}/{len(layers)}...')
                    scale, zero, g_idx = gptq[name].fasterquant(
                        percdamp=self.quantize_config.damp_percent,
                        groupsize=self.quantize_config.group_size,
                        actorder=self.quantize_config.desc_act
                    )
                    quantizers[f'{self.layers_block_name}.{i}.{name}'] = (
                        gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu()
                    )
                    gptq[name].free()

            for j in range(num_examples):
                layer_input = layer_inputs[j].unsqueeze(0).to(CUDA)
                layer_attention_mask = attention_masks[j].to(CUDA)
                layer_output = layer(layer_input, attention_mask=layer_attention_mask)[0][0].cpu()
                layer_outputs.append(layer_output)

            layers[i] = layer.to(CPU)
            del layer
            del gptq
            torch.cuda.empty_cache()

            layer_inputs, layer_outputs = layer_outputs, []

        pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size
        )
        self._quantized = True
        self.model.config.use_cache = forward_pass_use_cache

    def generate(self, inputs, **kwargs):
        """shortcut for model.generate"""
        return self.model.generate(inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_quantized(self, save_dir: str, use_safetensors: bool = False):
        """save quantized model and configs to local disk"""
        if not self.quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")

        self.model.to(CPU)

        model_save_name = f"gptq_model-{self.quantize_config.bits}bit"
        if use_safetensors:
            model_save_name += ".safetensors"
            state_dict = self.model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            safe_save(state_dict, join(save_dir, model_save_name))
        else:
            model_save_name += ".bin"
            torch.save(self.model.state_dict(), join(save_dir, model_save_name))

        self.model.config.save_pretrained(save_dir)
        self.quantize_config.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        bf16: bool = False,
        **model_init_kwargs
    ):
        """load un-quantized pretrained model to cpu"""

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        config = AutoConfig.from_pretrained(model_init_kwargs["pretrained_model_name_or_path"])
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        # enforce some values despite user specified
        model_init_kwargs["device_map"] = None
        model_init_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
        model_init_kwargs["low_cpu_mem_usage"] = False

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)
        model.seqlen = model.config.max_position_embeddings
        model.eval()

        return cls(model, False, quantize_config)

    @classmethod
    def from_quantized(
        cls,
        save_dir: str,
        device: str = "cpu",
        use_safetensors: bool = False
    ):
        """load quantized model from local disk"""
        config = AutoConfig.from_pretrained(save_dir)
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        quantize_config = BaseQuantizeConfig.from_pretrained(save_dir)

        model_save_name = f"gptq_model-{quantize_config.bits}bit"
        if use_safetensors:
            model_save_name += ".safetensors"
        else:
            model_save_name += ".bin"

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config, **{"low_cpu_mem_usage": False, "device_map": None})
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, quantize_config.bits, quantize_config.group_size)

        if model_save_name.endswith('.safetensors'):
            model.load_state_dict(safe_load(model_save_name, "cpu"))
        else:
            model.load_state_dict(torch.load(model_save_name))
        model.seqlen = model.config.max_position_embeddings

        model.eval()
        model.to(device)

        return model


__all__ = ["BaseGPTQForCausalLM", "BaseQuantizeConfig"]
