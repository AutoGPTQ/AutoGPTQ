from functools import partial
from typing import Union, Callable, Optional

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion
from torch.nn import functional as F


def act_dropout(
    hidden_states: torch.Tensor,
    activation: Union[Callable, nn.Module],
    dropout: float = 0.0
):
    hidden_states = activation(hidden_states)
    return hidden_states if dropout == 0.0 else F.dropout(hidden_states, dropout)


def dropout_res(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    dropout: float = 0.0
):
    hidden_states = hidden_states if dropout == 0.0 else F.dropout(hidden_states, dropout)
    return torch.add(hidden_states, residual)


def act_dropout_res(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    activation: Union[Callable, nn.Module],
    dropout: float = 0.0
):
    hidden_states = activation(hidden_states)
    hidden_states = hidden_states if dropout == 0.0 else F.dropout(hidden_states, dropout)
    return torch.add(hidden_states, residual)


class NVFusedActDropoutRes(nn.Module):
    def __init__(
        self,
        activation: Optional[Union[Callable, nn.Module]] = None,
        dropout: float = 0.0,
        residual: bool = False
    ):
        super(NVFusedActDropoutRes, self).__init__()

        fn = partial(F.dropout, p=dropout)
        if activation is not None and residual:
            fn = partial(act_dropout_res, activation=activation, dropout=dropout)
        elif activation is not None:
            fn = partial(act_dropout, activation=activation, dropout=dropout)
        elif residual:
            fn = partial(dropout_res, dropout=dropout)

        self.fn = fn
        self.residual = residual

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if isinstance(self.fn, nn.Dropout):
            return self.fn(hidden_states)

        inputs = {"hidden_states": hidden_states}
        if residual is not None and self.residual:
            inputs["residual"] = residual

        if hidden_states.device.type != "cuda":
            return self.fn(**inputs)

        aot_fn = memory_efficient_fusion(self.fn)
        return aot_fn(**inputs)


class FusedMLP(nn.Module):
    def __init__(
        self,
        input_proj: nn.Linear,
        out_proj: nn.Linear,
        activation: Optional[Union[Callable, nn.Module]] = None,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
        training: bool = False,
        residual: bool = False
    ):
        super(FusedMLP, self).__init__()

        if activation is None:
            activation = nn.Identity()

        self.input_proj = input_proj
        self.fused_op1 = NVFusedActDropoutRes(
            activation=activation,
            dropout=in_dropout if training else 0.0,
            residual=False
        )
        self.out_proj = out_proj
        self.fused_op2 = NVFusedActDropoutRes(
            activation=None,
            dropout=out_dropout if training else 0.0,
            residual=residual
        )

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        return self.fused_op2(self.out_proj(self.fused_op1(self.input_proj(hidden_states))), residual)


def gated_act_dropout(
    gate_states: torch.Tensor,
    up_states: torch.Tensor,
    activation: Union[Callable, nn.Module],
    dropout: float = 0.0
):
    hidden_states = activation(gate_states) * up_states
    return hidden_states if dropout == 0.0 else F.dropout(hidden_states, dropout)


class NVFusedGatedActDropout(nn.Module):
    def __init__(
        self,
        activation: Optional[Union[Callable, nn.Module]] = None,
        dropout: float = 0.0,
    ):
        super(NVFusedGatedActDropout, self).__init__()

        fn = partial(F.dropout, p=dropout)
        if activation is not None:
            fn = partial(gated_act_dropout, activation=activation, dropout=dropout)

        self.fn = fn

    def forward(self, gate_states: torch.Tensor, up_states):
        if isinstance(self.fn, nn.Dropout):
            return self.fn(gate_states * up_states)

        if gate_states.device.type != "cuda":
            return self.fn(gate_states, up_states)

        aot_fn = memory_efficient_fusion(self.fn)
        return aot_fn(gate_states, up_states)


class FusedGatedMLP(nn.Module):
    def __init__(
        self,
        input_proj: nn.Linear,
        out_proj: nn.Linear,
        activation: Optional[Union[Callable, nn.Module]] = None,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
        training: bool = False
    ):
        super(FusedGatedMLP, self).__init__()

        if activation is None:
            activation = nn.Identity()

        self.input_proj = input_proj
        self.fused_op = NVFusedGatedActDropout(activation=activation, dropout=in_dropout if training else 0.0)
        self.out_proj = out_proj
        self.out_dropout = nn.Dropout(out_dropout)

        self.intermediate_size = self.input_proj.out_features // 2

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.input_proj(hidden_states)
        return self.out_dropout(self.out_proj(self.fused_op(*hidden_states.chunk(chunks=2, dim=-1))))
