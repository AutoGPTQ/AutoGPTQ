from logging import getLogger

import torch.nn as nn
from transformers import AutoConfig

from ._const import SUPPORTED_MODELS


logger = getLogger(__name__)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_module_by_name(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def make_quant(module, names, bits, groupsize, name='', use_triton=False):
    if use_triton:
        raise NotImplementedError("triton not supported yet")
    else:
        from ..nn_modules.qlinear import QuantLinear

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


def pack_model(model, quantizers, bits, group_size, use_triton=False):
    if use_triton:
        raise NotImplementedError("triton not supported yet.")
    else:
        from ..nn_modules.qlinear import QuantLinear

    model.cpu()
    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size, use_triton=use_triton)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    logger.info('Model packed.')


def check_and_get_model_type(model_dir):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


__all__ = ["find_layers", "get_module_by_name", "make_quant", "pack_model", "check_and_get_model_type"]
