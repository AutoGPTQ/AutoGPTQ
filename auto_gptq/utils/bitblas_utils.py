import gc
import os
from logging import getLogger

import accelerate
import huggingface_hub
import torch
from accelerate.utils import find_tied_parameters
from safetensors.torch import save_file as safe_save
from tqdm import tqdm

from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear as BitBLASQuantLinear
from ..quantization import CHECKPOINT_FORMAT, QUANT_METHOD, BaseQuantizeConfig
from .import_utils import BITBLAS_AVAILABLE
from .modeling_utils import recurse_getattr, recurse_setattr


if BITBLAS_AVAILABLE:
    import bitblas

logger = getLogger(__name__)


def prepare_model_for_bitblas_load(
    model,
    quantize_config,
    quant_linear_class,
    torch_dtype,
    current_model_save_name,
    device_map,
):
    # The model (e.g. model.safetensors) is already serialized in the BitBLAS format, load it directly.
    if quantize_config.checkpoint_format == CHECKPOINT_FORMAT.BITBLAS:
        # if the checkpoint is already in bitblas format, we can load it directly.
        model_save_name = current_model_save_name
        logger.info(f"Loading a GPTQ model, detected BitBLAS serialized format at {model_save_name}.")
        model = convert_to_bitblas(model, quant_linear_class, quantize_config, repack=False)
    else:
        # otherwise, we need to convert the model to bitblas format first and cache locally from a gptq quant linear.
        model_save_name, is_cached = quantize_config.get_cache_file_path(quant_method=QUANT_METHOD.GPTQ,
                                                              checkpoint_format=CHECKPOINT_FORMAT.BITBLAS)

        # If GPTQ model has BitBLAS version cached locally, load from the cached version (no repacking needed).
        if is_cached:
            logger.info(
                f"Loading a GPTQ model, detected a cached repacked weight for BitBLAS kernel at {model_save_name}."
            )
            model = convert_to_bitblas(model, quant_linear_class, quantize_config, repack=False)

        # Otherwise, convert the model to BitBLAS format first and cache locally.
        else:
            # Loading the GPTQ checkpoint to do the conversion.
            # TODO: Avoid loading the model with wrong QuantLinear, and directly use
            # BitBLAS ones. The repacking can be done directly on the safetensors, just
            # as for AWQ checkpoints.
            accelerate.utils.modeling.load_checkpoint_in_model(
                model,
                dtype=torch_dtype,
                checkpoint=current_model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )
            # Convert model to bitblas, repacking weights into BitBLAS format.
            model = convert_to_bitblas(model, quant_linear_class, quantize_config, repack=True)

            # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
            tied_params = find_tied_parameters(model)

            for weight_group in tied_params:
                for param_name in weight_group:
                    if isinstance(recurse_getattr(model, param_name), torch.nn.Parameter):
                        recurse_setattr(
                            model,
                            param_name,
                            torch.nn.Parameter(recurse_getattr(model, param_name).clone()),
                        )
                    else:
                        recurse_setattr(
                            model,
                            param_name,
                            recurse_getattr(model, param_name).clone(),
                        )

            # Cache the converted model.
            safe_save(model.state_dict(), model_save_name)

    return model, model_save_name


def _get_cached_bitblas_save_name(model_name_or_path):
    """
    Gets The Cached Weight Path.
    If remote:   $HF_HOME/assets/autogptq/{model_name_or_path}/autogptq_model.safetensors
    If local:    {model_name_or_path}/autogptq_model.safetensors
    """
    if os.path.isdir(model_name_or_path):
        return os.path.join(model_name_or_path, "autogptq_model.safetensors")
    else:
        namespace, subfolder = model_name_or_path.split("/")
        assets_path = huggingface_hub.cached_assets_path(
            library_name="auto_gptq", namespace=namespace, subfolder=subfolder
        )
        return os.path.join(assets_path, "autogptq_model.safetensors")


@torch.no_grad()
def convert_to_bitblas(model, model_quantlinear, quantization_config, repack: bool, strict: bool = False):
    """
    Converts GPTQ-packed weights to the Bitblas format.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the BitBLAS's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with BitBLAS kernel..."
    else:
        # TODO: load directly BitBLAS QuantLinear.
        message = "Overriding QuantLinear layers to use BitBLAS's QuantLinear..."

    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when loading weights
        # from checkpoints holding zero bias.
        with torch.device("meta"):
            bitblas_module = BitBLASQuantLinear(
                bits=quantization_config.bits,
                group_size=quantization_config.group_size,
                infeatures=module.infeatures,
                outfeatures=module.outfeatures,
                bias=module.bias is not None,
                trainable=False,
                enable_tuning=True
            )

        # Dequantize the weight.
        if repack:
            bitblas_module.repack_from_gptq(module)

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, bitblas_module)

        # Free cuda memory.
        del module
        gc.collect()

    # Set quantization config to be BitBLAS.
    quantization_config.is_bitblas_format = True

    return model
