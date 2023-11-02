import torch
import torch.nn as nn

from typing import Optional, Union

def set_module_tensor_to_device_patched(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for 8 bit model serialization.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    if value is not None:
        if old_value.shape != value.shape:
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this look incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    device_quantization = None
    with torch.no_grad():
        # leave it on cpu first before moving them to cuda
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        if (
            param is not None
            and param.device.type != "cuda"
            and torch.device(device).type == "cuda"
            and param_cls.__name__ in ["Int8Params", "FP4Params"]
        ):
            device_quantization = device
            device = "cpu"
        if value is None:
            new_value = old_value.to(device)
            if device in ["meta", torch.device("meta")]:
                if dtype is None:
                    # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
                    new_value = new_value.to(old_value.dtype)
                elif not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)
                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ["Int8Params", "FP4Params"]:
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    # downcast to fp16 if any - needed for 8bit serialization
                    new_value = new_value.to(torch.float16)
                # quantize module that are going to stay on the cpu so that we offload quantized weights
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                setattr(module._parameters[tensor_name], "SCB", fp16_statistics.to(device))
                del fp16_statistics
            # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
            if (
                module.__class__.__name__ == "Linear8bitLt"
                and getattr(module.weight, "SCB", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "SCB", None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != "meta":
                        # if a bias exists, we need to wait until the bias is set on the correct device
                        module = module.cuda(device_index)
                    elif module.bias is None:
                        # if no bias exists, we can quantize right away
                        module = module.cuda(device_index)
            elif module.__class__.__name__ == "Linear4bit" and getattr(module.weight, "quant_state", None) is None:
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "quant_state", None) and device_index is not None:
                    module.weight = module.weight.cuda(device_index)
    # clean pre and post foward hook
    torch.cuda.empty_cache()
