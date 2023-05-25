import warnings
import re
from dataclasses import asdict
from enum import Enum

import torch
from auto_gptq.modeling._base import BaseGPTQForCausalLM, GeneralQuantLinear
from peft import get_peft_model, PeftConfig, PeftModel, TaskType, PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora import LoraLayer, LoraModel, Embedding, mark_only_lora_as_trainable, _freeze_adapter
from peft.utils.other import (
    _get_submodules,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
)


class QuantLoraLinear(torch.nn.Linear, LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        quant_linear_module: GeneralQuantLinear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        torch.nn.Linear.__init__(self, quant_linear_module.in_features, quant_linear_module.out_features)
        LoraLayer.__init__(self, quant_linear_module.in_features, quant_linear_module.out_features)

        self.quant_linear_module = quant_linear_module

        self.weight.requires_grad = False
        self.weight = self.quant_linear_module.weight
        self.bias = self.quant_linear_module.bias
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            torch.nn.init.ones_(self.lora_A[adapter_name].weight)
            torch.nn.init.zeros_(self.lora_B[adapter_name].weight)

    def merge(self):
        raise NotImplementedError("gptq model not support merge lora adapter")

    def unmerge(self):
        raise NotImplementedError("gptq model not support unmerge lora adapter")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return self.quant_linear_module(x)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self.quant_linear_module(x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = self.quant_linear_module(x)

            lora_B = self.lora_B[self.active_adapter]
            lora_A = self.lora_A[self.active_adapter]
            lora_dropout = self.lora_dropout[self.active_adapter]
            scale = self.scaling[self.active_adapter]

            x = x.type_as(lora_A.weight.data)
            adapter_result = (lora_B(lora_A(lora_dropout(x))) * scale).type_as(result)
            result += adapter_result
        else:
            result = self.quant_linear_module(x)

        result = result.to(previous_dtype)

        return result


class QuantLoraModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = False
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        if isinstance(target, GeneralQuantLinear):
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `GeneralQuantLinear` are supported."
                            )
                        new_module = QuantLoraLinear(adapter_name, target, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if not isinstance(new_module, QuantLoraLinear):
            new_module.weight = old_module.weight
            if hasattr(old_module, "bias"):
                if old_module.bias is not None:
                    new_module.bias = old_module.bias

            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        raise NotImplementedError("gptq model not support merge lora adapter")

    def unmerge_adapter(self):
        raise NotImplementedError("gptq model not support unmerge lora adapter")

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        raise NotImplementedError("gptq model not support merge and unload")

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight


def get_gptq_peft_model(
    model: BaseGPTQForCausalLM,
    peft_config: PeftConfig = None,
    model_id: str = None,
    adapter_name: str = "default"
):
    if not peft_config.task_type == TaskType.CAUSAL_LM:
        raise TypeError("only support CAUSAL_LM task type.")

    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = QuantLoraModel

    try:
        if model_id is None:
            if not peft_config:
                raise ValueError("peft_config can't be None when model_id is None.")
            peft_model = get_peft_model(model.model, peft_config)
        else:
            peft_model = PeftModel.from_pretrained(model.model, model_id, adapter_name)
    except:
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = LoraModel
        raise NotImplementedError(f"auto_gptq not support {peft_config.peft_type.value} peft type yet.")
    finally:
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = LoraModel

    return peft_model


__all__ = ["get_gptq_peft_model"]
