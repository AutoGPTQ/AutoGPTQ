from logging import getLogger
from typing import Optional

import torch
from packaging.version import parse as parse_version


try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import autogptq_cuda_64  # noqa: F401

    AUTOGPTQ_CUDA_AVAILABLE = True
except Exception:
    AUTOGPTQ_CUDA_AVAILABLE = False


try:
    import exllama_kernels  # noqa: F401

    EXLLAMA_KERNELS_AVAILABLE = True
except Exception:
    EXLLAMA_KERNELS_AVAILABLE = False

try:
    import exllamav2_kernels  # noqa: F401

    EXLLAMAV2_KERNELS_AVAILABLE = True
except Exception:
    EXLLAMAV2_KERNELS_AVAILABLE = False

try:
    import autogptq_marlin_cuda  # noqa: F401

    MARLIN_AVAILABLE = True
    MARLIN_EXCEPTION = None
except Exception as e:
    MARLIN_AVAILABLE = False
    MARLIN_EXCEPTION = e


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
        elif bits == 4 and not disable_exllamav2 and EXLLAMAV2_KERNELS_AVAILABLE:
            from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
        elif bits == 4 and not disable_exllama and EXLLAMA_KERNELS_AVAILABLE:
            from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear


def compare_transformers_version(version: str = "v4.28.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from transformers import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))


def compare_pytorch_version(version: str = "v2.0.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from torch import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))
