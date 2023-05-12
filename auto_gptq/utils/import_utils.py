def dynamically_import_QuantLinear(use_triton: bool, desc_act: bool, group_size: int):
    if use_triton:
        from ..nn_modules.qlinear_triton import QuantLinear
    else:
        if not desc_act or group_size == -1:
            from ..nn_modules.qlinear_old import QuantLinear
        else:
            from ..nn_modules.qlinear import QuantLinear

    return QuantLinear
