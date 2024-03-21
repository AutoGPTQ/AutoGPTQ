# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn


logger = getLogger(__name__)

try:
    import bitblas
except ImportError as e:
    bitblas_import_exception = e

    def error_raiser_bitblas(*args, **kwargs):
        raise ValueError(
            f"Trying to use the bitblas backend, but could not import dependencies with the following error: {bitblas_import_exception}"
        )

    autogptq_bitblas_cuda = bitblas_import_exception

from .qlinear_cuda_old import QuantLinear as QuantLinearOld
from bitblas.quantization.utils import general_compress, interleave_weight
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantizeConfig,
    MatmulWeightOnlyDequantize,
)
from bitblas.utils import get_target_from_env
from bitblas.cache import global_operator_cache
from typing import List, Union

BITBLAS_TARGET = get_target_from_env()
BITBLAS_DATABASE_PATH = ".bitblas_database"
BITBLAS_PROPAGATE_WEIGHTS = True
global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)


def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    return unpacked_zeros + 1


class QuantLinear(nn.Module):
    QUANT_TYPE = "bitblas"
    SUPPORTED_BITS = [1, 2, 4]
    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]
    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(
        self,
        bits: int,
        group_size: int,
        infeatures: int,
        outfeatures: int,
        bias: bool,
        enable_tuning: bool = True,
        fast_decoding: bool = True,
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        layout: str = "nt",
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._validate_parameters(bits, group_size, infeatures, outfeatures, trainable)

        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = self._set_group_size(group_size, infeatures)
        self.opt_features = opt_features
        self.target = BITBLAS_TARGET
        self._configure_bitblas_matmul(
            enable_tuning, fast_decoding, bias, propagate_b, layout, bits
        )
        self._initialize_buffers(infeatures, outfeatures, bias)

        self.reset_parameters()

    def _validate_parameters(
        self, bits, group_size, infeatures, outfeatures, trainable
    ):
        if infeatures % 16 != 0 or outfeatures % 16 != 0:
            raise ValueError("`infeatures` and `outfeatures` must be divisible by 16.")
        if bits not in self.SUPPORTED_BITS:
            raise NotImplementedError("Only 1/2/4 bits are supported.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")
        if trainable:
            raise NotImplementedError("Training is not supported.")

    def _set_group_size(self, group_size, infeatures):
        return infeatures if group_size == -1 else group_size

    def _initialize_buffers(self, infeatures, outfeatures, bias):
        self.register_buffer(
            "qweight",
            torch.empty(
                self.bitblas_matmul.retrieve_weight_shape(),
                dtype=self.TORCH_STORAGE_DTYPE,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (outfeatures, infeatures // self.group_size), dtype=self.TORCH_DTYPE
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (outfeatures, infeatures // self.group_size), dtype=self.TORCH_DTYPE
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=self.TORCH_DTYPE)
            )
        else:
            self.bias = None

    def _configure_bitblas_matmul(
        self, enable_tuning, fast_decoding, bias, propagate_b, layout, bits
    ):
        # Assuming MatmulWeightOnlyDequantizeConfig and MatmulWeightOnlyDequantize are defined elsewhere
        bitblas_dtype = self.BITBLAS_DTYPES[self.TORCH_DTYPE]
        matmul_config = MatmulWeightOnlyDequantizeConfig(
            M=self.opt_features,
            N=self.outfeatures,
            K=self.infeatures,
            in_dtype=bitblas_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            bit=bits,
            storage_dtype=self.STORAGE_DTYPE,
            source_format="uint",
            with_scaling=True,
            with_zeros=True,
            group_size=self.group_size,
            fast_decoding=fast_decoding,
            with_bias=bias,
            propagate_a=False,
            propagate_b=propagate_b,
            layout=layout,
            zeros_type="original",
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = MatmulWeightOnlyDequantize(config, target=self.target)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                print(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def reset_parameters(self):
        # init for char
        self.qweight = torch.randint_like(
            self.qweight,
            0,
            2 ** (self.bits - 1) - 1,
            dtype=torch.int8,
            device=self.qweight.device,
        )
        nn.init.normal_(self.scales)
        nn.init.zeros_(self.zeros)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def post_init(self):
        # TODO(lei): eliminate runtime overhead like exllama state
        pass

    def repack_from_gptq(self, gptq_module: QuantLinearOld):
        # qweight in gptq old quant linear stored with (outfeatures, infeatures), should be transposed.
        qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(qweight.cpu()).cuda()
        self.qweight = qweight
        # scales in gptq old quant linear stored with (infeatures // group_size, outfeatures), should be transposed.
        scales = gptq_module.scales.T.contiguous().view(self.TORCH_DTYPE)
        self.scales = scales
        # qzeros should be dequantized to int zeros.
        intzeros = unpack_qzeros(gptq_module.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_type == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_type == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_type}"
            )
        if self.bias is not None:
            self.bias = gptq_module.bias.data.to(torch.float16).contiguous()

    def forward(self, A):

        C = torch.empty(
            A.shape[:-1] + (self.scales.shape[0],), dtype=A.dtype, device=A.device
        )
        if self.bias is not None:
            self.bitblas_matmul(
                A.view((-1, A.shape[-1])),
                self.qweight,
                self.scales,
                self.zeros,
                self.bias,
                C.view((-1, C.shape[-1])),
            )
        else:
            self.bitblas_matmul(
                A.view((-1, A.shape[-1])),
                self.qweight,
                self.scales,
                self.zeros,
                C.view((-1, C.shape[-1])),
            )
        return C


__all__ = ["QuantLinear"]
