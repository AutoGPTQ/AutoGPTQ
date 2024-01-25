import torch
import huggingface_hub
import accelerate
import torch
from safetensors.torch import save_file as safe_save
from ..nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_marlin import dequantize_weight

import gc, os, copy
from logging import getLogger
from tqdm import tqdm

logger = getLogger(__name__)

def prepare_model_for_marlin_load(
    model_name_or_path, 
    model,
    quantize_config,
    quant_linear_class,
    torch_dtype,
    current_model_save_name,
    device_map
):  
     # If GPTQ model is serialized in the Marlin format, load directly (no repacking).
    if is_marlin_serialized(quantize_config):
        model_save_name = current_model_save_name
        logger.info(f"Loading a GPTQ model, detected Marlin serialized format at {model_save_name}.")
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=False)

    # If GPTQ model has Marlin version cached locally, load from the cached version (no repacking).
    elif is_marlin_cached(model_name_or_path):
        model_save_name = _get_cached_marlin_save_name(model_name_or_path)
        logger.info(f"Loading a GPTQ model, detected a cached repacked weight for Marlin kernel at {model_save_name}.")
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=False)
    
    # Otherwise, convert the model to Marlin format first and cache locally.
    else:
        # Loading the GPTQ checkpoint to do the conversion.
        # TODO: Avoid loading the model with wrong QuantLinear, and directly use
        # Marlin ones. The repacking can be done directly on the safetensors, just
        # as for AWQ checkpoints.
        model_save_name = current_model_save_name
        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True
        )
        # Convert model to marlin, repacking weights into Marlin format.
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=True)
        
        # Cache the converted model.
        model_save_name = cache_marlin(model, model_name_or_path)

    return model, model_save_name

# Gets The Cached Weight Path.
#   -- if remote:   $HF_HOME/assets/autogptq/{model_path}/autogptq_model.safetensors
#   -- if local:    {path}/autogptq_model.safetensors
def _get_cached_marlin_save_name(model_name_or_path):
    if os.path.isdir(model_name_or_path):
        return os.path.join(model_name_or_path, "autogptq_model.safetensors")
    else:
        namespace, subfolder = model_name_or_path.split("/")
        assets_path = huggingface_hub.cached_assets_path(library_name="autogptq", namespace=namespace, subfolder=subfolder)
        return os.path.join(assets_path, "autogptq_model.safetensors")

# Gets The Serialized Path.
#   -- if remote:   $HF_HOME/hub/models--
#   -- if local:    {path}
def _get_serialized_marlin_save_name(model_name_or_path):
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    else:
        namespace, subfolder = model_name_or_path.split("/")
        return huggingface_hub.cached_assets_path(library_name="hub", namespace=namespace, subfolder=subfolder) 

# Checks if a model stub is a marlin serialized model.
def is_marlin_serialized(quantize_config):
    if not hasattr(quantize_config, "is_marlin_format"):
        return False
    return quantize_config.is_marlin_format

# Checks if a model stub has a cached marlin version:
#   -- if remote:   $HF_HOME/assets/autogptq/{model_stub}.
#   -- if local:    {path}/autogptq_model.safetensors
def is_marlin_cached(model_name_or_path):
    model_save_name = _get_cached_marlin_save_name(model_name_or_path)
    return os.path.isfile(model_save_name)

# Caches Marlin model by saving autogptq_model.safetensors to
#   -- if remote:   $HF_HOME/assets/autogptq/{model_stub}.
#   -- if local:    {path}/autogptq_model.safetensors 
def cache_marlin(model, model_name_or_path):
    model_save_name = _get_cached_marlin_save_name(model_name_or_path)
    safe_save(model.state_dict(), model_save_name)
    return model_save_name

# Adapted from https://github.com/rib-2/marlin/tree/conversion
def _validate_marlin_compatibility(quantization_config):
    if quantization_config.bits != 4:
        return f"The quantized model uses a bitwidth different than 4 (found {quantization_config.bits})"
    if quantization_config.group_size != 128 and quantization_config.group_size != -1:
        return f"The quantized model uses a group size that is not 128 or -1 (found quantization_config.group_size)"
    if not quantization_config.sym:
        return f"The quantized model uses asymmetric quantization"
    if quantization_config.desc_act:
        return f"The quantized model uses act-order (also called desc-act) scheme"
    return None

@torch.no_grad()
def convert_to_marlin(model, model_quantlinear, quantization_config, repack: bool):
    """
    Converts GPTQ-packed weights to the Marlin format. This assumes that the model already meets Marlin kernel constraints.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the Marlin's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with Marlin kernel..."
    else:
        message = "Overriding QuantLinear layers to use Marlin's QuantLinear..."

    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        if module.bias is not None and torch.count_nonzero(module.bias) > 0:
            bias = module.bias
        else:
            bias = None

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1:]

        # Dequantize the weight.
        if repack:
            dequantized_weight, dequantized_qzeros = dequantize_weight(module)
            dequantized_weight = dequantized_weight.to(torch.float16)

            if not torch.all(dequantized_qzeros == 8):
                raise ValueError(f"Marlin kernel is compatible only with checkpoints using symetric quantization. Found non-symmetric quantization for the weight {name}.")

            linear_module = torch.nn.Linear(
                in_features=dequantized_weight.shape[1],
                out_features=dequantized_weight.shape[0],
                bias=bias is not None,
                dtype=torch.float16,
                device="cuda"
            )
            linear_module.weight.data.copy_(dequantized_weight)

            if bias is not None:
                linear_module.bias.data.copy_(bias)

            in_features = linear_module.in_features
            out_features = linear_module.out_features
        else:
            linear_module = torch.nn.Linear(
                in_features=module.infeatures, 
                out_features=module.outfeatures, 
                bias=bias is not None, 
                dtype=torch.float16, 
                device="cuda"
            )

        # Create new linear method and copy to model.
        new_module = MarlinQuantLinear(
            bits=4,
            group_size=module.group_size,
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            bias=bias is not None,
            trainable=False,
        )

        if repack:
            new_module.pack(linear_module, scales=copy.deepcopy(module.scales.data.t()).to("cuda"))

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del module
        if repack:
            del dequantized_weight
        torch.cuda.empty_cache()
        gc.collect()
    return model
