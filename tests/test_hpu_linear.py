import numpy as np
import math
import torch
import pytest
try:
    import habana_frameworks.torch.core as htcore
except Exception as e:
    pytestmark = pytest.mark.skip("Couldn't import HPU plugin, skipping HPU tests")

def _convert_to_tensor_list(tensor_or_tensors):
    if isinstance(tensor_or_tensors, tuple):
        return list(tensor_or_tensors)
    elif isinstance(tensor_or_tensors, list):
        return tensor_or_tensors
    elif isinstance(tensor_or_tensors, torch.Tensor):
        # You can't return list(tensor_or_tensors), because it will fail on 0-d tensors
        result_list = []
        result_list.append(tensor_or_tensors)
        return result_list
    else:
        raise TypeError("Can not convert outputs")

def compare_tensors(hpu_tensors, cpu_tensors, atol, rtol, assert_enable=True):
    hpu_tensors = _convert_to_tensor_list(hpu_tensors)
    cpu_tensors = _convert_to_tensor_list(cpu_tensors)
    assert len(hpu_tensors) == len(cpu_tensors)

    hpu_tensors = [tensor.to('cpu') if tensor is not None else tensor for tensor in hpu_tensors]

    for i in range(len(hpu_tensors)):
        if cpu_tensors[i] is None and hpu_tensors[i] is None:
            continue

        hpu_tensors[i] = (
            hpu_tensors[i].float()
            if hpu_tensors[i].dtype in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
            else hpu_tensors[i]
        )
        cpu_tensors[i] = (
            cpu_tensors[i].float()
            if cpu_tensors[i].dtype in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
            else cpu_tensors[i]
        )
        if assert_enable:
            np.testing.assert_allclose(
                hpu_tensors[i].detach().numpy(),
                cpu_tensors[i].detach().numpy(),
                atol=atol,
                rtol=rtol,
            )
        else:
            print("hpu_result[{}]".format(i), hpu_tensors[i].detach().numpy())
            print("cpu_result[{}]".format(i), cpu_tensors[i].detach().numpy())
            return np.allclose(
                hpu_tensors[i].detach().numpy(),
                cpu_tensors[i].detach().numpy(),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )

# taken from AutoGPTQ/tests/test_repacking.py
def gen_quant4(k, n, groupsize=-1, bias=False):
    maxq = 2 ** 4 - 1
    w = torch.randn((k, n), dtype=torch.bfloat16, device="cpu")

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
    ref = (w - (maxq + 1) // 2).bfloat16() * s

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = torch.nn.Linear(k, n, bias=bias)
    linear.weight.data = ref.t()

    return original_w, linear, s

@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [16, 32, 128])
@pytest.mark.parametrize("infeatures", [64, 128, 512, 4096, 11008])
@pytest.mark.parametrize("outfeatures", [64, 128, 512, 4096, 11008])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("scales_value, weight_value, zeros_value", [("normal", "normal", "normal"), ("normal", "normal", "range"), ("normal", "normal", "zeros"), ("ones", "zeros", "zeros"), ("ones", "zeros", "eights"), ("ones", "range", "zeros"), ("ones", "range", "ones"), ("ones", "7", "ones"), ("ones", "zeros", "range"),("ones", "zeros", "ones"), ("ones", "range", "range"), ("range", "range", "range"), ("range", "range", "zeros")])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_qlinear_hpu(bits, group_size, infeatures, outfeatures, bias, scales_value, weight_value, zeros_value, weight_dtype):
    qweight_shape_0 = infeatures // 32 * bits
    qzeros_shape_0 = math.ceil(infeatures / group_size)
    qzeros_shape_1 = outfeatures // 32 * bits
    if qweight_shape_0 == 0 or qzeros_shape_0 == 0 or qzeros_shape_1 == 0:
        pytest.skip(f"{qweight_shape_0=} == 0 or {qzeros_shape_0=} == 0 or {qzeros_shape_1=} == 0")
    if infeatures < group_size:
        pytest.skip(f"{infeatures=} < {group_size=}")
    if infeatures != outfeatures:
        pytest.skip(f"{infeatures=} != {outfeatures=}")
    trainable = False
    use_cuda_fp16 = False
    kernel_switch_threshold = 128
    from auto_gptq.nn_modules.qlinear import qlinear_hpu, qlinear_cuda_old
    quant_hpu = qlinear_hpu.QuantLinear(bits=bits, group_size=group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, use_cuda_fp16=use_cuda_fp16, kernel_switch_threshold=kernel_switch_threshold, trainable=trainable, weight_dtype=weight_dtype).to("hpu")
    # Cuda old implementation is the reference, also runs on hpu
    quant_ref_cuda_old = qlinear_cuda_old.QuantLinear(bits=bits, group_size=group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, use_cuda_fp16=use_cuda_fp16, kernel_switch_threshold=kernel_switch_threshold, trainable=trainable, weight_dtype=weight_dtype).to("hpu")
    input = torch.rand((infeatures, outfeatures), dtype=weight_dtype).to("hpu")
    _, linear, s = gen_quant4(infeatures, outfeatures, group_size, bias)

    if scales_value == "ones":
        s = torch.ones_like(s)
    if scales_value == "range":
        range_t = torch.tensor(list(range(1, s.numel()+1)), dtype=torch.int32)
        shape_s = s.shape
        s = (torch.ones(s.numel()) * range_t).reshape(shape_s).contiguous()

    if weight_value == "ones":
        linear.weight = torch.nn.Parameter(torch.ones_like(linear.weight))
    elif weight_value == "zeros":
        linear.weight = torch.nn.Parameter(torch.zeros_like(linear.weight))
    elif weight_value == "range":
        shape_w = linear.weight.shape
        weight_local = torch.ones(shape_w, dtype=torch.int32)
        range_t_weight = torch.tensor(list(range(0, 8)), dtype=torch.int32)
        linear.weight = torch.nn.Parameter((torch.ones(weight_local.numel(), dtype=linear.weight.dtype).reshape(-1, 8) * range_t_weight).reshape(shape_w).contiguous())
    elif weight_value.isnumeric():
        linear.weight = torch.nn.Parameter(torch.full_like(linear.weight, int(weight_value)))
    linear.weight = torch.nn.Parameter(linear.weight.to(weight_dtype))

    if zeros_value == "zeros":
        zeros = torch.full((infeatures // group_size, outfeatures), 0, dtype=torch.int32)
    elif zeros_value == "range":
        zeros = torch.ones((infeatures // group_size, outfeatures), dtype=torch.int32)
        range_t_zeros = torch.tensor(list(range(1, 9)), dtype=torch.int32)
        shape_z = zeros.shape
        zeros = (torch.ones(zeros.numel(), dtype=torch.int32).reshape(-1, 8) * range_t_zeros).reshape(shape_z).contiguous()
    elif zeros_value == "eights":
        zeros = torch.full((infeatures // group_size, outfeatures), 8, dtype=torch.int32)
    else:
        zeros = torch.full((infeatures // group_size, outfeatures), 1, dtype=torch.int32)

    htcore.mark_step()

    quant_ref_cuda_old.pack(linear, s.clone().detach().T, zeros.clone().detach().T, g_idx=None)
    htcore.mark_step()
    quant_ref_cuda_old.to("hpu")

    #TODO: pack independently
    quant_hpu.set_packed(quant_ref_cuda_old)
    htcore.mark_step()
    quant_hpu.to("hpu")

    out_ref_cuda_old = quant_ref_cuda_old(input)
    htcore.mark_step()
    quant_hpu.post_init()
    htcore.mark_step()
    out_hpu = quant_hpu(input)
    htcore.mark_step()

    out_ref_cuda_old = out_ref_cuda_old.cpu()
    out_hpu = out_hpu.cpu()
    compare_tensors(out_hpu.cpu(), out_ref_cuda_old.cpu(), rtol = 1e-05, atol = 1e-08)
