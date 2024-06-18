from logging import getLogger
from typing import Optional

import torch

logger = getLogger(__name__)


def dynamically_import_QuantLinear(
    use_triton: bool,
    desc_act: bool,
    group_size: int,
    bits: int,
    disable_exllama: Optional[bool] = None,
    disable_exllamav2: bool = False,
    use_marlin: bool = False,
):
    if use_triton:
        if torch.version.hip:
            logger.warning(
                "Running GPTQ triton version on AMD GPUs is untested and may result in errors or wrong predictions. Please use use_triton=False."
            )

        logger.debug("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
        if disable_exllama is None:
            if disable_exllamav2:
                disable_exllama = False
            else:
                disable_exllama = True
        if bits == 4 and use_marlin:
            from ..nn_modules.qlinear.qlinear_marlin import QuantLinear
        elif bits == 4 and not disable_exllamav2:
            from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
        elif bits == 4 and not disable_exllama:
            from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear
