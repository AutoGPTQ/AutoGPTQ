from packaging.version import parse as parse_version

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import autogptq_cuda

    AUTOGPTQ_CUDA_AVAILABLE = True
except:
    AUTOGPTQ_CUDA_AVAILABLE = False


def dynamically_import_QuantLinear(use_triton: bool, desc_act: bool, group_size: int, bits: int, disable_exllama: bool = False):
    if use_triton:
        from ..nn_modules.qlinear.qlinear_triton import QuantLinear
    else:
        if bits == 4 and not disable_exllama:
            from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear


def compare_transformers_version(
    version: str = "v4.28.0",
    op: str = "eq"
):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from transformers import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))


def compare_pytorch_version(
    version: str = "v2.0.0",
    op: str = "eq"
):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from torch import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))
