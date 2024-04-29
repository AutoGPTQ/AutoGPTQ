import itertools
import numpy as np
import random
import unittest

import torch
from torch import nn

from auto_gptq.nn_modules.qlinear.qlinear_qbits import (
    BITS_DTYPE_MAPPING,
    QuantLinear,
    convert_dtype_torch2str,
    dequantize_weight,
    unpack_to_8bit_signed
)
from intel_extension_for_transformers import qbits
from parameterized import parameterized


in_features = 256
out_features = 128
m = 16
group_size = 32
torch_dtype = torch.bfloat16 if qbits.check_isa_supported("AMX") else torch.float32

BITS = (
    4,
    8,
)
SYM = (
    True,
    False
)


def gen_quant(k, n, bits, groupsize=-1, sym=False):
    maxq = 2 ** bits - 1
    w = torch.randn((k, n), dtype=torch_dtype, device="cpu")

    original_w = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    if sym:
        s = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s *= 2 / maxq
        zeros = torch.full_like(s, (maxq + 1) // 2, dtype=torch.int32)
    else:
        xmax = w.max(0)[0]
        xmin = w.min(0)[0]
        range_width_channel = xmax - xmin
        s = range_width_channel / maxq
        # Compute the zero point
        zeros = torch.round(-xmin / s).to(dtype=torch.int32)

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - zeros).to(dtype=torch_dtype) * s

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    zeros = zeros.reshape((-1, n)).contiguous()

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s, zeros


class TestArcWeightOnly(unittest.TestCase):
    @parameterized.expand(list(itertools.product(BITS, SYM)))
    def test_quantization_4_cpu(self, bits, sym):
        random.seed(10)
        np.random.seed(10)
        torch.random.manual_seed(10)
        with torch.no_grad():
            _, linear, scales, qzeros = gen_quant(in_features, out_features, bits, group_size, sym=sym)
            qbits_linear = QuantLinear(bits=bits, group_size=group_size, infeatures=in_features, outfeatures=out_features, bias=False)
            qbits_linear.pack(linear, scales.T, qzeros.T, g_idx=None)
            fp_weight, _ = dequantize_weight(qbits_linear.qweight, qbits_linear.qzeros, qbits_linear.scales, bits)
            intweight, zeros = unpack_to_8bit_signed(qbits_linear.qweight, qbits_linear.qzeros, bits)

            if sym:
                self.assertIsNone(zeros)
            if zeros is None:
                zeros = torch.empty(0, dtype=torch.int8)
            else:
                # change it to int8 with offset 128
                if bits == 8:
                    zeros = (zeros.to(torch.int32) - (2 ** (bits - 1))).to(torch.int8)
                else:
                    zeros -= (2**(bits - 1))
                    zeros.bitwise_left_shift_(8 - bits)

            if sym:
                intweight -= (2**(bits - 1))
            intweight = intweight.to(torch.int8 if sym else torch.uint8)
            # due to asym return torch.uint8 but backend request int8,
            # change it to int8 with offset 128
            if not sym:
                intweight = (intweight.to(torch.int32) - (2 ** (bits - 1))).to(torch.int8)

            if bits == 4:
                intweight.bitwise_left_shift_((8 - bits))

            scales = scales if bits == 8 else scales / (2 ** (8 - bits))

            g_idx = torch.empty(0, dtype=torch.int32)
            qbits_qweight = qbits.repack_quantized_weight(intweight.contiguous(), scales.float(), zeros.contiguous(), g_idx,
                                                          BITS_DTYPE_MAPPING[bits],  # weight_dtype
                                                          "fp32",                    # scale_dtype
                                                          convert_dtype_torch2str(torch_dtype),  #compute_dtype
                                                          not sym,
                                                          group_size)
            qbits_out = torch.zeros(in_features, out_features, dtype=torch.float32)
            qbits.dequantize_packed_weight(
                qbits_qweight, qbits_out, False,
                convert_dtype_torch2str(torch_dtype),
                BITS_DTYPE_MAPPING[bits],
                "fp32")
            qbits_out = qbits_out.to(dtype=torch_dtype)
            assert (torch.allclose(qbits_out, fp_weight, rtol=0.0001))

            input = torch.rand(m, in_features, dtype=torch_dtype)
            qbit_input = input.clone()
            torch_out = torch.matmul(input, qbits_out)

            qbits_dst = torch.zeros(m, out_features, dtype=torch_dtype)
            qbits.woq_linear(
                qbit_input.contiguous(), qbits_qweight, torch.empty(0), qbits_dst, convert_dtype_torch2str(torch_dtype), "int4_clip", "fp32", not sym)

            print(abs((torch_out - qbits_dst)).max())
            assert (torch.allclose(qbits_dst, torch_out, atol=0.005 if torch_dtype == torch.float32 else 0.25))

if __name__ == "__main__":
    unittest.main()
