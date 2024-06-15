import copy
import unittest

import autogptq_marlin_cuda
import torch
import torch.nn as nn

from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear
from auto_gptq.nn_modules.qlinear.qlinear_marlin import _get_perms, dequantize_weight


def gen_quant4(k, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq + 1) // 2).half() * s

    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s


class TestRepacking(unittest.TestCase):
    def test_marlin_fast_repacking(self):
        k = 2048
        n = 1024
        m = 5
        group_size = 128

        _, linear, s = gen_quant4(k, n, group_size)
        cuda_old_linear = CudaOldQuantLinear(bits=4, group_size=group_size, infeatures=k, outfeatures=n, bias=False)

        zeros = torch.full((k // group_size, n), 8, dtype=torch.int32)

        cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

        # Adapted from utils.marlin_utils.convert_to_marlin
        dequantized_weight, dequantized_qzeros = dequantize_weight(cuda_old_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.all(dequantized_qzeros == 8))

        linear_module = torch.nn.Linear(
            in_features=k,
            out_features=n,
            bias=False,
            dtype=torch.float16,
            device="cuda",
        )
        linear_module.weight.data.copy_(linear.weight.data)  # Not using dequantized_weight to avoid approx

        # Create new linear method and copy to model.
        marlin_linear = MarlinQuantLinear(
            bits=4,
            group_size=group_size,
            infeatures=k,
            outfeatures=n,
            bias=False,
            trainable=False,
        )

        marlin_linear.pack(linear_module.to("cuda"), scales=copy.deepcopy(cuda_old_linear.scales.data.t()).to("cuda"))

        inp = torch.rand(m, k, dtype=torch.float16, device="cuda")

        cuda_old_linear = cuda_old_linear.to("cuda")
        marlin_linear = marlin_linear.to("cuda")
        with torch.no_grad():
            res_cuda_old = cuda_old_linear(inp)
            res_marlin = marlin_linear(inp)

        reldiff = (res_cuda_old - res_marlin).abs() / (res_cuda_old.abs() + 1e-12)
        self.assertTrue(torch.mean(reldiff) < 4e-3)

        weight_repacked = autogptq_marlin_cuda.gptq_repack(cuda_old_linear.qweight)
        self.assertTrue(torch.allclose(weight_repacked, marlin_linear.B))

        _, _scale_perm, _scale_perm_single = _get_perms()

        s = cuda_old_linear.scales.data.clone()
        if group_size != k:
            s = s.reshape((1, -1))
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, n)).contiguous()

        self.assertTrue(torch.allclose(s, marlin_linear.s))
