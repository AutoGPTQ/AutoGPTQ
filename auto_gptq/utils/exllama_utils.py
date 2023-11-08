import gc
import torch

from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear

def exllama_set_max_input_length(model, max_input_length: int):
    """
    This method does not necessarily require `model` to inherit from BaseGPTQForCausalLM.

    When using the exllama backend with act-order, it is necessary to initialize a buffer that depends on the maximum expected input length. In case the
    default used (EXLLAMA_DEFAULT_MAX_INPUT_LENGTH) is too short, this method can be called to extend the buffer size without reloading the whole model.
    """

    # The import is set here to avoid a global import. Arguably this is quite ugly, it would be better to have lazy loading.
    from exllama_kernels import prepare_buffers, cleanup_buffers_cuda

    if not model.quantize_config.desc_act:
        raise ValueError("The method exllama_set_max_input_length should be called only when using the exllama backend **with act-order**.")
    
    uses_exllama = False
    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaQuantLinear):
            uses_exllama = True
    
    if not uses_exllama:
        raise ValueError(f"The function exllama_set_max_input_length was called, but the model (instance of {model.__class__.__name__}) does not use the exllama backend for GPTQ. An other implementation is used (exllamav2, cuda, cuda-old, triton) and that the call to exllama_set_max_input_length is unnecessary. Please remove the call to exllama_set_max_input_length or use the exllama v1 backend.")

    device_to_buffers_size = {}
    for device, buffers in model.device_to_buffers.items():
        device_to_buffers_size[device] = {"max_dq_buffer_size": buffers["max_dq_buffer_size"], "max_inner_outer_dim": buffers["max_inner_outer_dim"]}
    
    # For an unknown reason calling just `del model.device_to_buffers` raises an AttributeError.
    for key in list(model.device_to_buffers.keys()):
        del model.device_to_buffers[key]
    model.device_to_buffers = None
    del model.device_to_buffers

    gc.collect()
    torch.cuda.empty_cache()
    cleanup_buffers_cuda()

    device_to_buffers = {}
    for device, buffers_size in device_to_buffers_size.items():
        # The temp_state buffer is required to reorder X in the act-order case.
        # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
        device_to_buffers[device] = {
            "temp_state": torch.zeros((max_input_length, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
            "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
            "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
            "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
        }

        prepare_buffers(device, device_to_buffers[device]["temp_state"], device_to_buffers[device]["temp_dq"])

    # Buffers need to be persistent to avoid any bug.
    model.device_to_buffers = device_to_buffers

    return model
