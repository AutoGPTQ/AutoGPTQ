from logging import getLogger

import torch.nn as nn
from transformers import AutoConfig

from ._const import SUPPORTED_MODELS
from ..quantization import make_quant, QuantLinear

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


def pack_model(model, quantizers, bits, group_size):
    model.cpu()
    logger.info('Packing model...')
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, bits, group_size)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    logger.info('Model packed.')


def check_and_get_model_type(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


__all__ = ["find_layers", "get_module_by_name", "pack_model", "check_and_get_model_type"]
