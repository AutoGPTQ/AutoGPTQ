import warnings
import re
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union, Tuple

import torch
from peft import get_peft_model, PeftConfig, PeftModel, PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora import LoraConfig, LoraLayer, LoraModel, Embedding
from peft.tuners.adalora import AdaLoraConfig, AdaLoraLayer, AdaLoraModel
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils.other import _get_submodules

from ..modeling._base import BaseGPTQForCausalLM
from ..nn_modules.qlinear import GeneralQuantLinear
from ..nn_modules.qlinear.qlinear_cuda import QuantLinear as QuantLinearCuda
from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear as QuantLinearCudaOld
from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as QuantLinearExllama
from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as QuantLinearExllamaV2
from ..nn_modules.qlinear.qlinear_qigen import QuantLinear as QuantLinearQigen
from ..nn_modules.qlinear.qlinear_triton import QuantLinear as QuantLinearTriton

LinearLayer = Union[torch.nn.Linear, GeneralQuantLinear, QuantLinearCuda,
                    QuantLinearCudaOld, QuantLinearExllama, QuantLinearExllamaV2, QuantLinearQigen,
                    QuantLinearTriton]

class GPTQLoraConfig(LoraConfig):
    injected_fused_attention: bool = False
    injected_fused_mlp: bool = False


def _get_linear_feature_count(linear_layer: LinearLayer) -> Tuple[int, int]:
    in_features = getattr(linear_layer, "in_features",
                          getattr(linear_layer, "infeatures"))
    out_features = getattr(linear_layer, "out_features",
                           getattr(linear_layer, "outfeatures"))
    return in_features, out_features


def _get_weight(linear_layer: LinearLayer) -> torch.Tensor:
    return getattr(linear_layer, "weight",
                   getattr(linear_layer, "qweight"))


class GPTQLoraLinear(torch.nn.Linear, LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        linear_module: LinearLayer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        in_features, out_features = _get_linear_feature_count(linear_module)
        torch.nn.Linear.__init__(self, in_features, out_features)
        LoraLayer.__init__(self, in_features, out_features)

        self.linear_module = linear_module

        delattr(self, "weight")
        self.weight = _get_weight(linear_module)
        delattr(self, "bias")

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            assert hasattr(linear_module, "weight")
            linear_module.weight.data = linear_module.weight.data.T

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
            torch.nn.init.zeros_(self.lora_B[adapter_name].weight)

    def merge(self):
        raise NotImplementedError("gptq model not support merge lora adapter")

    def unmerge(self):
        raise NotImplementedError("gptq model not support unmerge lora adapter")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return self.linear_module(x)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self.linear_module(x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = self.linear_module(x)

            lora_B = self.lora_B[self.active_adapter]
            lora_A = self.lora_A[self.active_adapter]
            lora_dropout = self.lora_dropout[self.active_adapter]
            scale = self.scaling[self.active_adapter]

            x = x.type_as(lora_A.weight.data)
            adapter_result = (lora_B(lora_A(lora_dropout(x))) * scale).type_as(result)
            result += adapter_result
        else:
            result = self.linear_module(x)

        result = result.to(previous_dtype)

        return result


class GPTQLoraModel(LoraModel):
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if not isinstance(new_module, GPTQLoraLinear):
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
                device = (
                    list(old_module.parameters()) + \
                    list(old_module.buffers())
                )[0].device
                module.to(device)

    @staticmethod
    def _create_new_module(lora_config: GPTQLoraConfig, adapter_name: str, target: torch.nn.Linear,
                           **kwargs):
        gptq_quantlinears = {
            GeneralQuantLinear, QuantLinearCuda,
            QuantLinearCudaOld, QuantLinearExllama, QuantLinearExllamaV2,
            QuantLinearQigen, QuantLinearTriton
        }

        is_gptq_layer = any([
            isinstance(target, cls)
            for cls in gptq_quantlinears
        ])
        if is_gptq_layer:
            return GPTQLoraLinear(
                adapter_name,
                target,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                fan_in_fan_out=lora_config.fan_in_fan_out,
            )
        else:
            return LoraModel._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )

    def merge_adapter(self):
        raise NotImplementedError("gptq model not support merge ada lora adapter")

    def unmerge_adapter(self):
        raise NotImplementedError("gptq model not support unmerge ada lora adapter")

    def merge_and_unload(self):
        raise NotImplementedError("gptq model not support merge and unload")


class GPTQAdaLoraConfig(AdaLoraConfig):
    injected_fused_attention: bool = False
    injected_fused_mlp: bool = False


class GPTQSVDLinear(torch.nn.Linear, AdaLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        linear_module: LinearLayer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        in_features, out_features = _get_linear_feature_count(linear_module)
        torch.nn.Linear.__init__(self, in_features, out_features)
        AdaLoraLayer.__init__(self, in_features, out_features)

        self.linear_module = linear_module

        delattr(self, "weight")
        self.weight = _get_weight(linear_module)
        delattr(self, "bias")
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            assert hasattr(linear_module, "weight")
            linear_module.weight.data = linear_module.weight.data.T

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        raise NotImplementedError("gptq model not support merge lora adapter")

    def unmerge(self):
        raise NotImplementedError("gptq model not support unmerge lora adapter")

    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return self.linear_module(x)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self.linear_module(x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = self.linear_module(x)
            result += (
                (
                    self.lora_dropout[self.active_adapter](x)
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                    @ self.lora_B[self.active_adapter].T
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
        else:
            result = self.linear_module(x)
        return result

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # Peft standard values seems too high
            # Still not ideal, just not causing NaNs with fp16 anymore
            torch.nn.init.normal_(self.lora_E[adapter_name], mean=0.0, std=0.005)
            torch.clamp_(self.lora_E[adapter_name].data, -0.1, 0.1)
            torch.nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.005)
            torch.clamp_(self.lora_A[adapter_name].data, -0.1, 0.1)
            torch.nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.005)
            torch.clamp_(self.lora_B[adapter_name].data, -0.1, 0.1)


class GPTQAdaLoraModel(AdaLoraModel):
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                device = (
                    list(old_module.parameters()) + \
                    list(old_module.buffers())
                )[0].device
                module.to(device)
    
    @staticmethod
    def _create_new_module(lora_config: GPTQLoraConfig, adapter_name: str, target: torch.nn.Linear,
                           **kwargs):
        gptq_quantlinears = {
            GeneralQuantLinear, QuantLinearCuda,
            QuantLinearCudaOld, QuantLinearExllama, QuantLinearExllamaV2,
            QuantLinearQigen, QuantLinearTriton
        }
        
        is_gptq_layer = any([
            isinstance(target, cls)
            for cls in gptq_quantlinears
        ])
        if is_gptq_layer:
            return GPTQSVDLinear(
                adapter_name,
                target,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                fan_in_fan_out=lora_config.fan_in_fan_out,
            )
        else:
            return LoraModel._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
    
    def merge_adapter(self):
        raise NotImplementedError("gptq model not support merge ada lora adapter")

    def unmerge_adapter(self):
        raise NotImplementedError("gptq model not support unmerge ada lora adapter")

    def merge_and_unload(self):
        raise NotImplementedError("gptq model not support merge and unload")


def find_all_linear_names(model: BaseGPTQForCausalLM, ignore: Optional[List[str]] = None, ignore_lm_head: bool = True):
    if not ignore:
        ignore = []
    lm_head_name = model.lm_head_name
    if ignore_lm_head and lm_head_name not in ignore:
        ignore.append(lm_head_name)
    results = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            res = n.split('.')[-1]
            if res not in ignore:
                results.add(res)
    return list(results)


@contextmanager
def hijack_peft_mappings():
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ADALORA] = GPTQAdaLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ADALORA] = GPTQAdaLoraModel

    try:
        yield
    except:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ADALORA] = GPTQAdaLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ADALORA] = GPTQAdaLoraModel
        raise
    finally:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ADALORA] = GPTQAdaLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ADALORA] = GPTQAdaLoraModel


def get_gptq_peft_model(
    model: BaseGPTQForCausalLM,
    peft_config: PeftConfig = None,
    model_id: str = None,
    adapter_name: str = "default",
    auto_find_all_linears: bool = True,
    train_mode: bool = False
):
    if train_mode and not model.trainable:
        model.enable_trainable_mode()
    if train_mode and not peft_config:
        raise ValueError("peft_config not specified when in train mode.")
    if not train_mode and not model_id:
        raise ValueError("model_id(where to load adapters) not specified when in inference mode.")

    if model.fused_attn_module_type is not None and not model.injected_fused_attention:
        peft_types = [PeftType.LORA.value, PeftType.ADALORA.value]
        warnings.warn(
            f"You can just ignore this warning if the peft type you use isn't in {peft_types}.\n"
            f"{model.__class__.__name__} supports injecting fused attention but not enables this time. "
            "If you are training adapters, you must also disable fused attention injection when loading quantized "
            "base model at inference time, otherwise adapters may not be added to base model properly. "
            "If you are loading adapters to do inference, you can reference to adapter's config file to check "
            "whether the adapters are trained using base model that not enable fused attention injection."
        )
    if model.injected_fused_mlp:
        raise NotImplementedError("GPTQ model that enables fused mlp injection is not supported to integrate with peft.")

    if train_mode:
        peft_type = peft_config.peft_type
        if not isinstance(peft_type, str):
            peft_type = peft_type.value
        if peft_type in [PeftType.LORA.value, PeftType.ADALORA.value]:
            if auto_find_all_linears:
                peft_config.target_modules = find_all_linear_names(model, ignore_lm_head=True)
            if peft_type == PeftType.LORA.value and not isinstance(peft_config, GPTQLoraConfig):
                peft_config = GPTQLoraConfig(**peft_config.to_dict())
            if peft_type == PeftType.ADALORA.value and not isinstance(peft_config, GPTQAdaLoraConfig):
                peft_config = GPTQAdaLoraConfig(**peft_config.to_dict())
            peft_config.injected_fused_attention = model.injected_fused_attention
            peft_config.injected_fused_mlp = model.injected_fused_mlp
        if peft_type == PeftType.ADAPTION_PROMPT.value:
            if peft_config.adapter_layers > model.config.num_hidden_layers:
                warnings.warn(
                    f"model has only {model.config.num_hidden_layers} layers "
                    f"but adapter_layers is set to {peft_config.adapter_layers}, "
                    f"will reset value to {model.config.num_hidden_layers}."
                )
                peft_config.adapter_layers = model.config.num_hidden_layers
            if model.injected_fused_attention:
                raise NotImplementedError(
                    "model with fused attention injected isn't supported to use ADAPTION_PROMPT peft type yet."
                )

    with hijack_peft_mappings():
        try:
            if train_mode:
                peft_model = get_peft_model(model.model, peft_config, adapter_name=adapter_name)
            else:
                peft_model = PeftModel.from_pretrained(model.model, model_id, adapter_name)
        except:
            raise
            raise NotImplementedError(
                f"{model.__class__.__name__} not support {peft_config.peft_type.value} peft type yet."
            )

    return peft_model


__all__ = [
    "GPTQLoraConfig",
    "GPTQLoraModel",
    "GPTQAdaLoraConfig",
    "GPTQAdaLoraModel",
    "find_all_linear_names",
    "get_gptq_peft_model"
]
