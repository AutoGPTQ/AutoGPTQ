from packaging.version import parse as parse_version


def dynamically_import_QuantLinear(use_triton: bool, desc_act: bool, group_size: int):
    if use_triton:
        from ..nn_modules.qlinear_triton import QuantLinear
    else:
        if not desc_act or group_size == -1:
            from ..nn_modules.qlinear_old import QuantLinear
        else:
            from ..nn_modules.qlinear import QuantLinear

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
