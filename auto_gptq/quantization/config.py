import json
import logging
import os
from dataclasses import dataclass, field, fields
from os.path import isdir, join
from typing import Optional

import huggingface_hub
from transformers.utils.hub import PushToHubMixin, cached_file


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)

CHECKPOINT_FORMAT_FIELD = "checkpoint_format"
CHECKPOINT_FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
QUANT_METHOD_FIELD = "quant_method"
QUANT_CONFIG_FILENAME = "quantize_config.json"


# checkpoint formats
class CHECKPOINT_FORMAT:
    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"
    MARLIN = "marlin"
    AWQ_GEMM = "gemm"


# quant methods
class QUANT_METHOD:
    GPTQ = "gptq"
    AWQ = "awq"


QUANT_METHOD_FORMAT_MAPPING = {
    QUANT_METHOD.GPTQ: {
        CHECKPOINT_FORMAT.GPTQ,
        CHECKPOINT_FORMAT.GPTQ_V2,
        CHECKPOINT_FORMAT.MARLIN,
    },
    QUANT_METHOD.AWQ: {
        CHECKPOINT_FORMAT.AWQ_GEMM
    }
}

# awq is inference only
QUANTIZE_BLACK_LIST = {QUANT_METHOD.AWQ}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": "bits",
    "q_group_size": "group_size",
}


@dataclass
class BaseQuantizeConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    quant_method: str = field(default=QUANT_METHOD.GPTQ)
    checkpoint_format: str = field(default=CHECKPOINT_FORMAT.GPTQ_V2)
    model_name_or_path: Optional[str] = field(default=None)
    model_file_base_name: Optional[str] = field(default=None)

    def __post_init__(self):
        fields_info = fields(self)

        # validate quant method and format is matched
        valid_checkpoint_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.quant_method, None)
        if valid_checkpoint_formats is None:
            raise ValueError(f"Unsupported quantization method: {self.quant_method}")

        if self.checkpoint_format not in valid_checkpoint_formats:
            raise ValueError(
                f"The checkpoint format used is {self.checkpoint_format}, and the quantization method is {self.quant_method}. "
                f"This is not supported, please open an issue at https://github.com/AutoGPTQ/AutoGPTQ/issues.")

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")

        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir,  QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    # normalize quant config for compat and also performs validation
    def from_quant_config(cls, quantize_cfg, checkpoint_format: str = None):
        valid_formats = {CHECKPOINT_FORMAT.GPTQ, CHECKPOINT_FORMAT.GPTQ_V2, CHECKPOINT_FORMAT.MARLIN, CHECKPOINT_FORMAT.AWQ_GEMM}

        checkpoint_format_auto_inferred = False
        # compat: checkpoint_format can be passed in via from_quantized() if field missing from json
        if checkpoint_format:
            if checkpoint_format not in valid_formats:
                raise ValueError(f"Unknown quantization checkpoint format: {checkpoint_format}.")
            if quantize_cfg.get(CHECKPOINT_FORMAT_FIELD):
                raise ValueError("Conflict: quantization checkpoint_format is passed in and also exists in model config.")
        # compat: warn if checkpoint_format is missing
        elif quantize_cfg.get(CHECKPOINT_FORMAT_FIELD) is None:
            checkpoint_format_auto_inferred = True

        field_names = [field.name for field in fields(cls)]

        normalized = {
            QUANT_METHOD_FIELD: QUANT_METHOD.GPTQ,
            # compat: default to gptq(v1) when loading models
            CHECKPOINT_FORMAT_FIELD: checkpoint_format if checkpoint_format else CHECKPOINT_FORMAT.GPTQ
        }
        for key, val in quantize_cfg.items():
            key = key.lower()

            # remap keys according to compat map
            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]

            if key == CHECKPOINT_FORMAT_FIELD:
                val = val.lower()

                if val in {CHECKPOINT_FORMAT.GPTQ, CHECKPOINT_FORMAT.GPTQ_V2, CHECKPOINT_FORMAT.MARLIN, CHECKPOINT_FORMAT.AWQ_GEMM}:
                    normalized[key] = val
                else:
                    raise ValueError(f"Unknown quantization format: {val}.")
            elif key == QUANT_METHOD_FIELD:
                val = val.lower()
                # compat: some hf models use quant_method=marlin
                if val == CHECKPOINT_FORMAT.MARLIN:
                    normalized[CHECKPOINT_FORMAT_FIELD] = CHECKPOINT_FORMAT.MARLIN
                elif val not in {QUANT_METHOD.GPTQ, QUANT_METHOD.AWQ}:
                    raise ValueError(f"Unknown quantization method: {val}.")
                else:
                    normalized[QUANT_METHOD_FIELD] = val
            elif key == CHECKPOINT_FORMAT_FIELD_COMPAT_MARLIN and val:
                normalized[CHECKPOINT_FORMAT_FIELD] = CHECKPOINT_FORMAT.MARLIN
            elif key == "version" and val.lower() == CHECKPOINT_FORMAT.AWQ_GEMM:
                normalized[QUANT_METHOD_FIELD] = QUANT_METHOD.AWQ
                normalized[CHECKPOINT_FORMAT_FIELD] = CHECKPOINT_FORMAT.AWQ_GEMM
            elif key in field_names:
                normalized[key] = val
            else:
                logger.info(f"Ignoring unknown parameter in the quantization configuration: {key}.")

        if checkpoint_format_auto_inferred:
            logger.info(f"`checkpoint_format` is missing from the quantization configuration and is automatically inferred to {normalized[CHECKPOINT_FORMAT_FIELD]}.")

        if normalized[CHECKPOINT_FORMAT_FIELD] in {CHECKPOINT_FORMAT.AWQ_GEMM, CHECKPOINT_FORMAT.MARLIN}:
            # AWQ and Marlin do not reorder the rows.
            normalized["desc_act"] = False

        if "sym" not in normalized:
            logger.warning(
                "The quantization configuration does not contain an entry `sym` (symmetric quantization). "
                "This may result in silent errors. Defaulting to `sym=True`."
            )

        return cls(**normalized)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        checkpoint_format = kwargs.pop("checkpoint_format", None)

        transformers_config = False
        for quantize_config_filename in [
            QUANT_CONFIG_FILENAME,
            "quant_config.json",
            "config.json",
        ]:
            if isdir(save_dir):  # Local
                resolved_config_file = join(save_dir, quantize_config_filename)
            else:  # Remote
                resolved_config_file = cached_file(
                    save_dir,
                    quantize_config_filename,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
            if resolved_config_file is not None:
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)

            if transformers_config:
                args_from_json = args_from_json["quantization_config"]

            return cls.from_quant_config(args_from_json, checkpoint_format)

    def get_cache_file_path(self, quant_method: QUANT_METHOD = None, checkpoint_format: CHECKPOINT_FORMAT = None):
        """
        Gets The Cached Weight Path.
        If remote:   $HF_HOME/assets/autogptq/{model_name_or_path}/_{quant-method}_{checkpoint_format}.safetensors
        If local:    {model_name_or_path}/autogptq_model_{quant-method}_{checkpoint_format}.safetensors
        """

        use_quant_method = quant_method if quant_method else self.quant_method
        use_checkpoint_format = checkpoint_format if checkpoint_format else self.checkpoint_format

        cache_file_name = f"autogptq_model_{use_quant_method}_{use_checkpoint_format}.safetensors"

        if os.path.isdir(self.model_name_or_path):
            cache_file_name = os.path.join(self.model_name_or_path, cache_file_name)
        else:
            namespace, subfolder = self.model_name_or_path.split("/")
            assets_path = huggingface_hub.cached_assets_path(
                library_name="auto_gptq", namespace=namespace, subfolder=subfolder
            )
            cache_file_name = os.path.join(assets_path, cache_file_name)

        return cache_file_name, os.path.isfile(cache_file_name)

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "static_groups": self.static_groups,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "model_name_or_path": self.model_name_or_path,
            "model_file_base_name": self.model_file_base_name,
            QUANT_METHOD_FIELD: self.quant_method,
            CHECKPOINT_FORMAT_FIELD: self.checkpoint_format,
        }
