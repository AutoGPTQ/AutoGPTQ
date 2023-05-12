from abc import abstractmethod

import torch.nn as nn


class FusedBaseModule(nn.Module):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs):
        raise NotImplementedError()


class FusedBaseAttentionModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, group_size=-1, use_cuda_fp16=True, desc_act=False, **kwargs):
        raise NotImplementedError()


class FusedBaseMLPModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        raise NotImplementedError()
