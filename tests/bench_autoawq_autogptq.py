import torch


try:
    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"AutoAWQ package (https://github.com/casper-hansen/AutoAWQ) is required to run this benchmark. {e}"
    )

import numpy as np

from auto_gptq.modeling._utils import autogptq_post_init
from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear


group_size = 128
bits = 4

# Yi 34B down_proj
k = 20480
n = 7168

device = torch.device("cuda:0")

linear_class = dynamically_import_QuantLinear(use_triton=False, desc_act=False, group_size=group_size, bits=4)

linear_gptq = linear_class(
    bits=bits,
    group_size=group_size,
    infeatures=k,
    outfeatures=n,
    bias=False,
)

assert isinstance(linear_gptq, QuantLinear)

linear_gptq = linear_gptq.eval()
linear_gptq = linear_gptq.to(device)

linear_gptq = autogptq_post_init(linear_gptq, use_act_order=False)

num_runs = 60

lines = []

seqlens = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    80,
    120,
    250,
    512,
    1024,
    2048,
    4000,
    8000,
]

print(f"in_features={k}, out_features={n}")
for query_length in seqlens:
    # batch_size, query_length, hidden_size
    inp = torch.rand(1, query_length, k, dtype=torch.float16).to(device)

    torch.cuda.empty_cache()

    # Warmup Exllama v2
    with torch.no_grad():
        res = linear_gptq(inp)

    latencies = []
    torch.cuda.synchronize()
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        res = linear_gptq(inp)

        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    # print("-------")
    # print(f"Latency GPTQ Exllama v2 (query_length={query_length}): {np.mean(latencies):.3f} ms, p10={np.percentile(latencies, 10):.3f}, p90={np.percentile(latencies, 90):.3f}")

    exllamav2_mean_latency = np.mean(latencies)
    exllamav2_p10 = np.percentile(latencies, 10)
    exllamav2_p90 = np.percentile(latencies, 90)

    torch.cuda.empty_cache()

    total_seqlen = inp.shape[:-1].numel()
    if total_seqlen <= 8:
        awq_kernel = "GEMV"
        linear_awq = WQLinear_GEMV(
            w_bit=bits,
            group_size=group_size,
            in_features=k,
            out_features=n,
            bias=False,
            dev=device,
        )
    else:
        awq_kernel = "GEMM"
        linear_awq = WQLinear_GEMM(
            w_bit=bits,
            group_size=group_size,
            in_features=k,
            out_features=n,
            bias=False,
            dev=device,
        )

    # Warmup AWQ
    with torch.no_grad():
        res = linear_awq(inp)

    latencies = []
    torch.cuda.synchronize()
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        res = linear_awq(inp)

        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    awq_mean_latency = np.mean(latencies)
    awq_p10 = np.percentile(latencies, 10)
    awq_p90 = np.percentile(latencies, 90)

    exllama_speedup = awq_mean_latency / exllamav2_mean_latency

    # print(f"Latency AWQ (query_length={query_length}, kernel={awq_kernel}): {np.mean(latencies):.3f} ms, p10={np.percentile(latencies, 10):.3f}, p90={np.percentile(latencies, 90):.3f}")

    line = "{},{},{},{},{},{},{},{},{},{},{}".format(
        bits,
        group_size,
        total_seqlen,
        awq_kernel,
        f"{awq_mean_latency:.3f}",
        f"{exllamav2_mean_latency:.3f}",
        f"{awq_p10:.3f}",
        f"{awq_p90:.3f}",
        f"{exllamav2_p10:.3f}",
        f"{exllamav2_p90:.3f}",
        f"{exllama_speedup:.3f}",
    )
    lines.append(line)


header = "bits, group_size, total_seqlen, awq_kernel, awq_mean_latency (ms), exllamav2_mean_latency (ms), awq_p10, awq_p90, exllamav2_p10, exllamav2_p90, exllama_speedup"

print(header)
for line in lines:
    print(line)
