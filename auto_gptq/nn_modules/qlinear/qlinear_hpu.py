import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
try:
    import habana_frameworks.torch.core as htcore
    convert_from_uint4 = torch.ops.hpu.convert_from_uint4
except Exception as e:
    hpu_import_exception = e

    def error_raiser_hpu(*args, **kwargs):
        raise ValueError(
            f"Trying to use HPU, but could not import the HPU framework with the following error: {hpu_import_exception}"
        )

    convert_from_uint4 = error_raiser_hpu


logger = getLogger(__name__)

def pack_tensor(input, bits = 4):
    normal = input.to(torch.int32)
    q = torch.zeros((normal.shape[0], normal.shape[1] // 32 * bits), dtype=torch.int32)
    i = 0
    col = 0
    while col < q.shape[1]:
        for j in range(i, i + (32 // bits)):
            q[:, col] |= normal[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    q = q.to(torch.int32)
    return q

class QuantLinear(nn.Module):
    QUANT_TYPE = "hpu"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        logger.debug(f"qlinear_hpu QuantLinear::__init__ {bits=}, {group_size=}, {infeatures=}, {outfeatures=}, {bias=}, {use_cuda_fp16=}, {kernel_switch_threshold=}, {trainable=}, {weight_dtype=}")
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

    def _preprocessing(self):
        self.qweight = self.qweight.cpu()
        weight = self.unpack_weight_from_cuda_old_format()
        new_qweight = pack_tensor(weight)
        self.qweight = new_qweight.to('hpu')

        # TODO: Support group indexing and remove the check
        columns = self.qweight.shape[0]
        g_idx_trivial = [i // self.group_size for i in range(columns)]
        g_idx_trivial = torch.tensor(g_idx_trivial, dtype=torch.int32)
        assert torch.equal(self.g_idx, g_idx_trivial), "Non-trivial tensor g_idx is not supported"

        zeros = self.unpack_zeros_from_cuda_old_format().cpu()
        new_qzeros = pack_tensor(zeros)
        self.qzeros = new_qzeros.to('hpu')

    def post_init(self):
        self._preprocessing()

    def pack(self, linear, scales, zeros, g_idx):
        #TODO: implement
        raise NotImplementedError("QuantLinear HPU currently doesn't support packing")

    def set_packed(self, qlinear_cls):
        self.qweight = qlinear_cls.qweight
        self.qzeros = qlinear_cls.qzeros
        self.scales = qlinear_cls.scales
        self.bias = qlinear_cls.bias

    def forward(self, x):
        x_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros
        weight = convert_from_uint4(qweight, scales, zeros, x_dtype)
        out = torch.matmul(x, weight)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    def unpack_zeros_from_cuda_old_format(self):
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        ).to(self.scales.dtype)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
        zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
        return zeros

    def unpack_weight_from_cuda_old_format(self):
        weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape((weight.shape[0]*weight.shape[1], weight.shape[2]))
        return weight

__all__ = ["QuantLinear"]
