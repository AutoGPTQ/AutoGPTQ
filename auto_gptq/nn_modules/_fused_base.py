from abc import abstractmethod
from logging import getLogger

import torch.nn as nn

logger = getLogger(__name__)

try:
    from .triton_utils import TritonModuleMixin
except ImportError:
    logger.error('triton not installed.')
    raise


class FusedBaseModule(nn.Module, TritonModuleMixin):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs):
        raise NotImplementedError()


class FusedBaseAttentionModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, group_size=-1, use_cuda_fp16=True, desc_act=False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class FusedBaseMLPModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        raise NotImplementedError()
