import copy
import logging
import os
from os.path import isdir, join
from typing import Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers
import threadpoolctl as tctl
from accelerate.hooks import remove_hook_from_module
from safetensors import safe_open
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import (
    CommitOperationAdd,
    PushToHubMixin,
    create_commit,
    create_repo,
)

from ..nn_modules.qlinear import GeneralQuantLinear
from ..quantization import GPTQ, BaseQuantizeConfig
from ..quantization.config import (
    META_FIELD_QUANTIZER,
    META_QUANTIZER_AUTOGPTQ,
    MIN_VERSION_WITH_V2,
    FORMAT,
    FORMAT_FIELD,
    QUANT_METHOD_FIELD,
    QUANTIZE_BLACK_LIST,
)
from ..utils.data_utils import collate_data
from ..utils.import_utils import (
    AUTOGPTQ_CUDA_AVAILABLE,
    EXLLAMA_KERNELS_AVAILABLE,
    EXLLAMAV2_KERNELS_AVAILABLE,
    MARLIN_AVAILABLE,
    TRITON_AVAILABLE,
    dynamically_import_QuantLinear,
)
from ..utils.marlin_utils import (
    _validate_marlin_compatibility,
    _validate_marlin_device_support,
    prepare_model_for_marlin_load,
)
from ..version import __version__
from ._const import CPU, CUDA_0, SUPPORTED_MODELS
from ._utils import (
    autogptq_post_init,
    convert_gptq_v1_to_v2_format,
    convert_gptq_v2_to_v1_format,
    find_layers,
    get_checkpoints,
    get_device,
    get_module_by_name_prefix,
    get_module_by_name_suffix,
    make_quant,
    make_sure_no_tensor_in_meta_device,
    move_to_device,
    pack_model,
    simple_dispatch_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v


class BaseGPTQForCausalLM(nn.Module, PushToHubMixin):
    layer_type: str = None
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None
    lm_head_name: str = "lm_head"

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: BaseQuantizeConfig,
        is_triton_backend: bool = False,
        trainable: bool = False,
        qlinear_kernel: nn.Module = None
    ):
        super().__init__()

        self.model = model
        self.model_type = self.model.config.model_type
        self._quantized = quantized
        self.quantize_config = quantize_config
        self.config = self.model.config

        self.is_triton_backend = is_triton_backend
        self.trainable = trainable

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        self.qlinear_kernel = qlinear_kernel

    @property
    def quantized(self):
        return self._quantized

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)

    def _prepare_examples_for_quantization(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
    ):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_examples = []
        for example in examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id

        new_examples = [
            collate_data(new_examples[start : start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples

    @torch.inference_mode()
    def quantize(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = True,
    ):
        if self.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}")

        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logger.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        examples = self._prepare_examples_for_quantization(examples, batch_size)

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = get_module_by_name_prefix(self.model, self.layers_block_name)

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if cache_examples_on_gpu else CPU
        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to_device(inp, data_device))
            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs["attention_mask"] is not None:
                attention_masks.append(kwargs["attention_mask"].to(data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to_device(pos_ids, data_device))
            one_kwargs = {}
            for (
                k,
                v,
            ) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to_device(v, data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == CPU:
            layers[0] = layers[0].to(CUDA_0)
            force_layer_back_to_cpu = True

        ori_outside_layer_module_devices = {}
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to_device(module, cur_layer_device)

        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for example in examples:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to_device(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        handle.remove()

        move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to_device(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        inside_layer_modules = self.inside_layer_modules
        if not self.quantize_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        quantizers = {}

        # stores all per-layer quant stats such as avg loss and processing time
        quant_log = []

        layer_count = len(layers)
        layer_pb = tqdm(range(layer_count))
        for i in layer_pb:
            layer_pb.set_description(f"Quantizing layer {i + 1} of {layer_count}")
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, CUDA_0)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = find_layers(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names if n in full}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        self.quantize_config.bits,
                        perchannel=True,
                        sym=self.quantize_config.sym,
                        mse=False,
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        # gptq is mutable.
                        gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = []
                    for k, layer_inp in enumerate(layer_inputs[j]):
                        layer_input.append(move_to_device(layer_inp, cur_layer_device))

                    layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                    additional_layer_inputs = {"attention_mask": layer_attention_mask}
                    layer_position_ids = (
                        None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                    )
                    if layer_position_ids is not None:
                        additional_layer_inputs["position_ids"] = layer_position_ids
                    for k, v in layer_input_kwargs[j].items():
                        additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
                    layer(*layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                for name in subset:
                    layer_pb.set_description(f"Quantizing {name} in layer {i + 1} of {layer_count}")

                    try:
                        scale, zero, g_idx, duration, avg_loss = gptq[name].fasterquant(
                            percdamp=self.quantize_config.damp_percent,
                            group_size=self.quantize_config.group_size,
                            actorder=self.quantize_config.desc_act,
                            static_groups=self.quantize_config.static_groups,
                        )

                        stat = {"layer": i + 1, "module": name, "avg_loss": avg_loss, "time": duration}

                        quant_log.append(stat)
                        logger.info(stat)

                    except torch._C._LinAlgError as e:
                        if  "not positive-definite" in str(e).lower():
                            logger.warning("Please increase damp or nsamples for calibration data to avoid the following quant error. "
                                  "Ref: https://github.com/AutoGPTQ/AutoGPTQ/issues/572#issuecomment-2006686913")
                        raise e

                    quantizers[f"{self.layers_block_name}.{i}.{name}"] = (
                        gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device),
                    )
                    gptq[name].free()

            for j in range(num_batches):
                layer_input = []
                for k, layer_inp in enumerate(layer_inputs[j]):
                    layer_input.append(move_to_device(layer_inp, cur_layer_device))

                layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
                layer_output = move_to_device(
                    layer(*layer_input, **additional_layer_inputs)[0],
                    cur_layer_device if cache_examples_on_gpu else CPU,
                )
                layer_outputs.append([layer_output])

            layers[i] = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []  # TODO: is it really OK to cache only the first positional argument?
            torch.cuda.empty_cache()

        logger.info(f"Quantization summary:\n{quant_log}")
        for module_log in quant_log:
            logger.info(module_log)

        self.qlinear_kernel = pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=self.quantize_config.desc_act,
            warmup_triton=autotune_warmup_after_quantized,
            force_layer_back_to_cpu=force_layer_back_to_cpu,
            use_marlin=self.quantize_config.format == FORMAT.MARLIN,
        )
        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = simple_dispatch_model(self.model, device_map)
        self.model.config.use_cache = forward_pass_use_cache

        self._quantized = True

        torch.cuda.empty_cache()

        return quant_log

    @property
    def device(self):
        if not self.hf_device_map:
            return self.model.device
        else:
            device = [d for d in self.hf_device_map.values() if d not in {"disk"}][0]
            return torch.device(device)

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_quantized(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        format: Optional[str] = None,
        use_safetensors: bool = True,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        # write autogptq tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=META_QUANTIZER_AUTOGPTQ,
            version=__version__,
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)
        quantize_config = copy.deepcopy(self.quantize_config)
        model = self.model

        if not self.quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")

        if format == FORMAT.GPTQ_V2 or (format is None and quantize_config.format == FORMAT.GPTQ_V2):
            logger.warning(
                f"Using 'checkpoint_format = {FORMAT.GPTQ_V2}': the serialized model is only supported by AutoGPTQ version >= {MIN_VERSION_WITH_V2}."
            )

        if format is not None and quantize_config.format != format:
            # Model qzeros may be edited in place.
            # TODO: avoid inplace modification of the weights
            model = copy.deepcopy(self.model)

            if format == FORMAT.GPTQ_V2:
                if quantize_config.format != FORMAT.GPTQ:
                    raise NotImplementedError(f"Asked to serialize a model with `checkpoint_format={format}` but the model format is {quantize_config.format}. This is not supported. Please open an issue at https://github.com/AutoGPTQ/AutoGPTQ/issues.")

                model = convert_gptq_v1_to_v2_format(
                    model,
                    quantize_config=quantize_config,
                    qlinear_kernel=self.qlinear_kernel,
                )

                quantize_config.format = FORMAT.GPTQ_V2
            elif format == FORMAT.GPTQ:
                if quantize_config.format != FORMAT.GPTQ_V2:
                    raise NotImplementedError(f"Asked to serialize a model with `checkpoint_format={format}` but the model format is {quantize_config.format}. This is not supported. Please open an issue at https://github.com/AutoGPTQ/AutoGPTQ/issues.")

                model = convert_gptq_v2_to_v1_format(
                    model,
                    quantize_config=quantize_config,
                    qlinear_kernel=self.qlinear_kernel
                )

                quantize_config.format = FORMAT.GPTQ

        # internal is always gptq v2 but allow users to pass gptq (v1) via config
        if format is None and quantize_config.format == FORMAT.GPTQ:
            # Model qzeros may be edited in place.
            # TODO: avoid inplace modification of the weights
            model = copy.deepcopy(self.model)
            model = convert_gptq_v2_to_v1_format(
                model,
                quantize_config=quantize_config,
                qlinear_kernel=self.qlinear_kernel
            )

        model.to(CPU)

        if quantize_config.model_file_base_name is None:
            if use_safetensors:
                model_base_name = "model"
            else:
                model_base_name = "pytorch_model"
        else:
            model_base_name = quantize_config.model_file_base_name

        if use_safetensors:
            model_save_name = model_base_name + ".safetensors"
            state_dict = model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            if safetensors_metadata is None:
                safetensors_metadata = {}
            elif not isinstance(safetensors_metadata, dict):
                raise TypeError("safetensors_metadata must be a dictionary.")
            else:
                logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                new_safetensors_metadata = {}
                converted_keys = False
                for key, value in safetensors_metadata.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        converted_keys = True
                        try:
                            new_key = str(key)
                            new_value = str(value)
                        except Exception as e:
                            raise TypeError(
                                f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                            )
                        if new_key in new_safetensors_metadata:
                            logger.warning(
                                f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                            )
                        new_safetensors_metadata[new_key] = new_value
                safetensors_metadata = new_safetensors_metadata
                if converted_keys:
                    logger.debug(
                        f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                    )

            # Format is required to enable Accelerate to load the metadata
            # otherwise it raises an OSError
            safetensors_metadata["format"] = "pt"
            safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
        else:
            model_save_name = model_base_name + ".bin"
            torch.save(model.state_dict(), join(save_dir, model_save_name))

        config.quantization_config = quantize_config.to_dict()
        config.save_pretrained(save_dir)

        quantize_config.model_name_or_path = save_dir
        quantize_config.model_file_base_name = model_base_name
        quantize_config.save_pretrained(save_dir)

    def save_pretrained(
        self,
        save_dir: str,
        **kwargs,
    ):
        logger.warning("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        **model_init_kwargs,
    ):
        """load un-quantized pretrained model to cpu"""

        if not torch.cuda.is_available():
            raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        # Parameters related to loading from Hugging Face Hub
        cache_dir = model_init_kwargs.pop("cache_dir", None)
        force_download = model_init_kwargs.pop("force_download", False)
        resume_download = model_init_kwargs.pop("resume_download", False)
        proxies = model_init_kwargs.pop("proxies", None)
        local_files_only = model_init_kwargs.pop("local_files_only", False)
        use_auth_token = model_init_kwargs.pop("use_auth_token", None)
        revision = model_init_kwargs.pop("revision", None)
        subfolder = model_init_kwargs.pop("subfolder", "")
        commit_hash = model_init_kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_commit_hash": commit_hash,
        }

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **cached_file_kwargs
        )
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype
        model_init_kwargs["trust_remote_code"] = trust_remote_code
        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.tie_weights()

            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
                low_zero=False,
            )
            model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
            )
            model_init_kwargs["low_cpu_mem_usage"] = True

            del model
        else:
            model_init_kwargs["device_map"] = None
            model_init_kwargs["low_cpu_mem_usage"] = False

        torch.cuda.empty_cache()

        merged_kwargs = {**model_init_kwargs, **cached_file_kwargs}
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **merged_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return cls(model, False, quantize_config)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        use_triton: bool = False,
        use_marlin: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        trainable: bool = False,
        disable_exllama: Optional[bool] = None,
        disable_exllamav2: bool = False,
        format: Optional[str|FORMAT] = None,
        **kwargs,
    ):
        """load quantized model from local disk"""
        # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
        if disable_exllama is None:
            if disable_exllamav2:
                disable_exllama = False
            else:
                disable_exllama = True

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        if use_triton and not TRITON_AVAILABLE:
            logger.warning("Triton is not installed, reset use_triton to False.")
            use_triton = False
        if not disable_exllama and not EXLLAMA_KERNELS_AVAILABLE:
            logger.warning(
                "Exllama kernel is not installed, reset disable_exllama to True. "
                "This may because you installed auto_gptq using a pre-build wheel "
                "on Windows, in which exllama_kernels are not compiled. To use "
                "exllama_kernels to further speedup inference, you can re-install "
                "auto_gptq from source."
            )
            disable_exllama = True
        if not disable_exllamav2 and not EXLLAMAV2_KERNELS_AVAILABLE:
            logger.warning(
                "Exllamav2 kernel is not installed, reset disable_exllamav2 to True. "
                "This may because you installed auto_gptq using a pre-build wheel "
                "on Windows, in which exllama_kernels are not compiled. To use "
                "exllama_kernels to further speedup inference, you can re-install "
                "auto_gptq from source."
            )
            disable_exllamav2 = True
        if not AUTOGPTQ_CUDA_AVAILABLE:
            logger.warning(
                "CUDA kernels for auto_gptq are not installed, this will result in "
                "very slow inference speed. This may because:\n"
                "1. You disabled CUDA extensions compilation by setting BUILD_CUDA_EXT=0 when install auto_gptq from source.\n"
                "2. You are using pytorch without CUDA support.\n"
                "3. CUDA and nvcc are not installed in your device."
            )

        if not disable_exllamav2 and not disable_exllama:
            logger.warning(
                "You have activated both exllama and exllamav2 kernel. Setting disable_exllama to True and keeping disable_exllamav2 to False"
            )
            disable_exllama = True

        # == step1: prepare configs and file names == #
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, format=format, **cached_file_kwargs, **kwargs)
        else:
            if not isinstance(quantize_config, BaseQuantizeConfig):
                quantize_config = BaseQuantizeConfig.from_quant_config(quantize_config, format)

        if quantize_config.format == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            use_marlin = True

        marlin_compatible = _validate_marlin_device_support()
        if use_marlin and not MARLIN_AVAILABLE:
            raise TypeError("use_marlin is true but Marlin is not available due to cuda/device support.")

        if not use_marlin and MARLIN_AVAILABLE:
            unsupported_reason = _validate_marlin_compatibility(quantize_config)
            if unsupported_reason is None and marlin_compatible:
                logger.info(
                    "You passed a model that is compatible with the Marlin int4*fp16 GPTQ kernel but use_marlin is False. We recommend using `use_marlin=True` to use the optimized Marlin kernels for inference. Example: `model = AutoGPTQForCausalLM.from_quantized(..., use_marlin=True)`."
                )

        if model_basename is None:
            if quantize_config.model_file_base_name:
                possible_model_basenames = [quantize_config.model_file_base_name]
            else:
                possible_model_basenames = [
                    f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
                    "model",
                ]
        else:
            possible_model_basenames = [model_basename]

        quantize_config.model_name_or_path = model_name_or_path

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)

        # Retrieve (and if necessary download) the quantized checkpoint(s).
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(model_name_or_path=model_name_or_path, extensions=extensions, possible_model_basenames=possible_model_basenames, **cached_file_kwargs)

        quantize_config.model_file_base_name = true_model_basename

        model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.

        if (not disable_exllama or not disable_exllamav2) and trainable:
            logger.warning(
                "QuantLinear with the exllama backend not does support the trainable mode yet, switching to cuda/cuda_old/triton backend."
            )
            disable_exllama = True
            disable_exllamav2 = True

        elif not use_triton and trainable:
            logger.warning(
                "QuantLinear with cuda backend not support trainable mode yet, Switch to the pytorch backend."
            )

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        if torch_dtype is None:
            torch_dtype = torch.float16

        if torch_dtype != torch.float16:
            logger.warning("Overriding use_cuda_fp16 to False since torch_dtype is not torch.float16.")
            use_cuda_fp16 = False


        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights(include_buffers=False))

        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )

            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == cls.lm_head_name:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                    not name.endswith(ignore_layer)
                    for sublist in cls.inside_layer_modules
                    for ignore_layer in sublist
                ):
                    logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act,
                trainable=trainable,
                use_marlin=quantize_config.format == FORMAT.MARLIN,
            )
            model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == "cuda" else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = accelerate.utils.get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type],
                    low_zero=(device_map == "balanced_low_0"),
                )
        if not isinstance(device_map, dict):
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
            )

        if low_cpu_mem_usage:
            make_sure_no_tensor_in_meta_device(
                model,
                use_triton,
                quantize_config.desc_act,
                quantize_config.group_size,
                bits=quantize_config.bits,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
            )

        if use_marlin:
            if is_sharded:
                raise ValueError("The loading of sharded checkpoints with Marlin is currently not supported. Please raise an issue in AutoGPTQ repository.")
            if torch.version.hip:
                raise ValueError("Can not use Marlin int4*fp16 kernel with AMD ROCm version of PyTorch as the kernel is not compatible. Please do not use `use_marlin=True` when using ROCm devices.")
            if not _validate_marlin_device_support():
                raise ValueError(f'Can not use Marlin int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel. Please do not use `use_marlin=True`, or please upgrade your GPU ("The more you buy, the more you save." - Taiwanese proverb).')

            # Validate the model can run in Marlin.
            if torch_dtype != torch.float16:
                raise ValueError("Marlin kernel requires torch_dtype=torch.float16.")
            unsupported_reason = _validate_marlin_compatibility(quantize_config)
            if unsupported_reason is not None:
                raise ValueError(
                    f"The model {model_name_or_path} can not be converted to use the Marlin kernel for the following reason: {unsupported_reason}, which is not supported by Marlin kernel."
                )

            # Load the quant linear type we need.
            # TODO: load marlin directly with the right quantlinear class.
            quant_linear_class = dynamically_import_QuantLinear(
                use_triton=use_triton,
                desc_act=quantize_config.desc_act,
                group_size=quantize_config.group_size,
                bits=quantize_config.bits,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_marlin=False,
            )

            # Prepare model for marlin load.
            #   If stub is marlin serialized         --> load from directly
            #   If stub has cached marlin version   --> load from the cached version
            #   Otherwise                           --> convert to marlin, cache, load from cache
            model, model_save_name = prepare_model_for_marlin_load(
                model=model,
                quantize_config=quantize_config,
                quant_linear_class=quant_linear_class,
                torch_dtype=torch_dtype,
                current_model_save_name=model_save_name,
                device_map=device_map,
            )

        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )

        # TODO: Why are we using this custom function and not dispatch_model?
        model = simple_dispatch_model(model, device_map)

        qlinear_kernel = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=quantize_config.desc_act,
            group_size=quantize_config.group_size,
            bits=quantize_config.bits,
            disable_exllama=disable_exllama,
            disable_exllamav2=disable_exllamav2,
            use_marlin=use_marlin,
        )

        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if quantize_config.format == FORMAT.GPTQ:
            # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
            if not quantize_config.sym and not quantize_config.is_quantized_or_packed_by_v2():
                raise ValueError(
                    f"Loading of a sym=False model with checkpoint_format={FORMAT.GPTQ} is only supported if produced by autogptq version >= {MIN_VERSION_WITH_V2}")

            logger.info(f"Compatibility: converting `checkpoint_format` from `{FORMAT.GPTQ}` to `{FORMAT.GPTQ_V2}`.")

            model = convert_gptq_v1_to_v2_format(
                model,
                quantize_config=quantize_config,
                qlinear_kernel=qlinear_kernel,
            )

            quantize_config.format = FORMAT.GPTQ_V2

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # Any post-initialization that require device information, for example buffers initialization on device.
        model = autogptq_post_init(model, use_act_order=quantize_config.desc_act)

        model.eval()

        # == step6: (optional) warmup triton == #
        if use_triton and warmup_triton:
            from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
            QuantLinear.warmup(model, seqlen=model.seqlen)


        # == step7: make model compatible with peft
        # cls.make_sure_compatible_with_peft(
        #     model,
        #     use_triton,
        #     quantize_config.desc_act,
        #     quantize_config.group_size,
        #     bits=quantize_config.bits,
        #     disable_exllama=disable_exllama,
        #     disable_exllamav2=disable_exllamav2,
        #     use_marlin=use_marlin,
        # )

        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton,
            trainable=trainable,
            qlinear_kernel=qlinear_kernel,
        )

    def warmup_triton(self, enabled: bool = True):
        if not enabled:
            return
        if not TRITON_AVAILABLE:
            logger.warning("triton is not available, skip warmup stage directly.")
            return

        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
        QuantLinear.warmup(self.model, seqlen=self.model.seqlen)

    def enable_trainable_mode(self, enabled: bool = True):
        if not self.is_triton_backend and enabled:
            raise NotImplementedError("For now, trainable mode only supports triton backend.")
        for n, m in self.model.named_modules():
            if hasattr(m, "trainable"):
                setattr(m, "trainable", enabled)

    def disable_trainable_mode(self):
        self.enable_trainable_mode(enabled=False)

    @staticmethod
    def make_sure_compatible_with_peft(
        model: PreTrainedModel,
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: bool = True,
        disable_exllamav2: bool = False,
        use_marlin: bool = False,
    ):
        GeneralQuantLinear.inject_to_model(
            model,
            dynamically_import_QuantLinear(use_triton, desc_act, group_size, bits=bits, disable_exllama=disable_exllama,
                                           disable_exllamav2=disable_exllamav2,
                                           use_marlin=use_marlin),
        )

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)


__all__ = ["BaseGPTQForCausalLM", "BaseQuantizeConfig"]
