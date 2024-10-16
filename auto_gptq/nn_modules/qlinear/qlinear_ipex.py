import math

import numpy as np
import torch
import torch.nn as nn
import transformers
from ...utils.import_utils import IPEX_AVAILABLE
from logging import getLogger


logger = getLogger(__name__)

if IPEX_AVAILABLE:
    from intel_extension_for_pytorch.nn.modules.weight_only_quantization import WeightOnlyQuantizedLinear

BITS_DTYPE_MAPPING = {
    8: "int8",
}


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


class QuantLinear(nn.Module):
    QUANT_TYPE = "ipex"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        if bits not in [4]:
            raise NotImplementedError("Only 4-bits is supported for IPEX.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.asym = True

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

        self.kernel_switch_threshold = kernel_switch_threshold

        self.trainable = trainable

        # for training forward
        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

    def post_init(self):
        assert self.qweight.device.type == "cpu"
        if not self.trainable:
            self.ipex_linear = WeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros, \
                                                                    self.infeatures, self.outfeatures, None, self.bias, \
                                                                    self.group_size, self.g_idx, 0, 0)

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x: torch.Tensor):
        if not self.trainable and hasattr(self, "ipex_linear"):
            return self.ipex_linear(x)

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights, dim=1)
        out = torch.matmul(x, weights)
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out

        return out


@torch.no_grad()
def unpack_to_8bit_signed(qweight, qzeros, bits, g_idx=None):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    zeros = None
    if not torch.all(torch.eq(qzeros, 2004318071 if bits == 4 else 0b01111111011111110111111101111111)):
        zp_shape = list(qzeros.shape)
        zp_shape[1] = zp_shape[1] * (32 // bits)

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
        if bits == 8:
            zeros = zeros.to(torch.uint8)
        zeros = zeros + 1
        try:
            zeros = zeros.reshape(zp_shape)
        except:
            # zeros and scales have different iteam numbers.
            # remove 1 (due to 0 + 1 in line 252)
            zeros = zeros[zeros != 1]
            zeros = zeros.reshape(zp_shape)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight.bitwise_and_((2**bits) - 1)
    weight = weight.view(-1, weight.shape[-1])

    if g_idx is not None:
        group_size = weight.shape[0] // qzeros.shape[0]
        weight2 = weight.clone()
        group_dict = {}
        for i in range(len(g_idx)):
            group_idx = g_idx[i].item()
            if group_idx not in group_dict:
                target_idx = group_idx * group_size
                group_dict[group_idx] = 0
            else:
                group_dict[group_idx] = group_dict[group_idx] + 1
                target_idx = group_idx * group_size + group_dict[group_idx]
            weight2[target_idx] = weight[i]
        weight = weight2

    return weight, zeros


# Copied from qlinear_marlin.py
@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales, bits):
    unpacked_qweight, unpacked_qzeros = unpack_to_8bit_signed(qweight, qzeros, bits)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    if unpacked_qzeros is not None:
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    else:
        unpacked_qzeros = torch.full_like(scales, 8 if bits == 4 else 128, dtype=torch.int32)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight, unpacked_qzeros


__all__ = ["QuantLinear"]