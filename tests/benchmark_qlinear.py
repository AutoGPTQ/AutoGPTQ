import argparse
import itertools
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton

from auto_gptq.modeling._utils import autogptq_post_init
from auto_gptq.nn_modules.qlinear import (
    qlinear_cuda,
    qlinear_cuda_old,
    qlinear_exllama,
    qlinear_exllamav2,
    qlinear_triton,
    qlinear_tritonv2,
)


FIXTURES_PATH = Path(__file__).parent.absolute() / "fixtures"
COLORS = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
STYLES = itertools.cycle(["solid", "dashed", "dashdot", "dotted"])


@dataclass
class GPTQState:
    infeatures: int
    outfeatures: int
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    # wf: torch.Tensor
    bits: int
    maxq: int
    group_size: int


def download_test_modules(save_dir):
    print("Downloading pre-quantized llama and mistral models...")
    from transformers import AutoModelForCausalLM

    dtype = torch.float16

    mistal_model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-v0.1-GPTQ",
        torch_dtype=dtype,
        device_map="auto",
    )
    llama_model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-GPTQ",
        torch_dtype=dtype,
        device_map="auto",
    )

    for model in [mistal_model, llama_model]:
        model_name = model.model.__class__.__name__
        torch.save(
            model.model.layers[0].self_attn,
            f"{save_dir}/gptq_{model_name}_self_attn0_layer0.pt",
        )
        torch.save(
            model.model.layers[0].mlp, f"{save_dir}/gptq_{model_name}_mlp_layer0.pt"
        )


def make_perf_report(
    x_names,
    x_vals,
    line_vals,
    title="Qlinear Bench",
    ylabel="ms",
    line_names=None,
    x_log=True,
):
    line_styles = [(next(COLORS), next(STYLES)) for _ in range(len(line_vals))]

    bench = triton.testing.Benchmark(
        x_names=x_names,  # Argument names to use as an x-axis for the plot.
        x_vals=x_vals,  # Different possible values for `x_name`.
        x_log=x_log,  # x axis is logarithmic.
        line_arg="kernel",
        # Argument name whose value corresponds to a different line in the plot.
        line_vals=line_vals,  # , "fast"],  # Possible values for `line_arg`.
        line_names=(
            line_names if line_names else line_vals
        ),  # , "fast"],  # Label name for the lines.
        styles=line_styles,  # Line styles.
        ylabel=ylabel,  # Label name for the y-axis.
        plot_name=title,  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
    return triton.testing.perf_report(bench)


def get_qstate(fixtures_path, model="llama", module="self_attn", layer="q_proj"):
    module_paths = os.listdir(fixtures_path)
    try:
        module_path = next(
            p for p in module_paths if (model in p.lower() and module in p.lower())
        )
    except ValueError:
        raise ValueError(f"Could not find {model} {module} in {module_paths}")
    print(f"Loading {module_path}")

    module = torch.load(os.path.join(fixtures_path, module_path))
    layer = getattr(module, layer)
    return GPTQState(
        infeatures=layer.infeatures,
        outfeatures=layer.outfeatures,
        qweight=layer.qweight.cuda(),
        qzeros=layer.qzeros.cuda(),
        scales=layer.scales.cuda(),
        g_idx=layer.g_idx.cuda(),
        bits=layer.bits,
        maxq=layer.maxq,
        group_size=layer.group_size,
    )


def make_data(batch_size, seqlen, hidden_size, dtype=torch.float16, seed=3407):
    torch.manual_seed(seed)
    X = torch.randn(batch_size, seqlen, hidden_size, dtype=dtype).cuda()
    return X


BENCHMARK_QLAYER_NAMES = [
    "qlinear_cuda_old",
    "qlinear_cuda",
    "qlinear_triton",
    "qlinear_tritonv2",
    "qlinear_exllama",
    "qlinear_exllamav2",
]
BENCHMARK_QLAYER_TYPES = [
    qlinear_cuda_old,
    qlinear_cuda,
    qlinear_triton,
    qlinear_tritonv2,
    qlinear_exllama,
    qlinear_exllamav2,
]


def get_qlayers(
    qstate: GPTQState,
    max_seq_len,
    layer_names=BENCHMARK_QLAYER_NAMES,
    layer_types=BENCHMARK_QLAYER_TYPES,
):
    qlayers = {}
    for linear_name, linear_cls in zip(layer_names, layer_types):
        linear = linear_cls.QuantLinear(
            bits=qstate.bits,
            group_size=qstate.group_size,
            infeatures=qstate.infeatures,
            outfeatures=qstate.outfeatures,
            bias=False,
        )
        linear.qweight = qstate.qweight
        linear.qzeros = qstate.qzeros
        linear.scales = qstate.scales
        linear.g_idx = qstate.g_idx
        linear.maxq = qstate.maxq
        if isinstance(linear, qlinear_exllama.QuantLinear) or isinstance(
            linear, qlinear_exllamav2.QuantLinear
        ):
            linear = autogptq_post_init(
                linear, use_act_order=False, max_input_length=max_seq_len
            )
        qlayers[linear_name] = linear

    return qlayers


def get_diff(a, ref):
    eps = 1e-6
    return f"Maxdiff: {(a - ref).abs().max()}, Mean relative diff: {((a - ref).abs() / (ref.abs() + eps)).mean()}"


def run_check(X, qlayers: dict):
    outs = {}
    for qname, qlayer in qlayers.items():
        outs[qname] = qlayer(X)
    ref = outs.pop("qlinear_cuda_old")

    for name, test_out in outs.items():
        print(f"qlinear_cuda_old vs {name}: {get_diff(ref, test_out)}")


def benchmark(seqlen, kernel, qstate: GPTQState = None, check=False):
    qlayers = get_qlayers(qstate, max_seq_len=seqlen)
    X = make_data(1, seqlen, qstate.infeatures, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if check:
        run_check(X, qlayers)

    quants = triton.testing.do_bench(lambda: qlayers[kernel](X), quantiles=quantiles)
    return quants


def run_bench(
    qstate: GPTQState,
    seqlen=[128, 256, 512, 1024, 2048],
    check=False,
    outdir="qlinear_bench",
):
    x_names = ["seqlen"]
    x_vals = seqlen
    line_vals = BENCHMARK_QLAYER_NAMES

    reporter = make_perf_report(x_names=x_names, x_vals=x_vals, line_vals=line_vals)
    bench = partial(benchmark, qstate=qstate, check=check)
    runner = reporter(bench)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    runner.run(print_data=True, save_path=outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--fixtures_path",
        type=str,
        default=FIXTURES_PATH,
        help="Path to saved GPTQ pre-quantized modules.  Defaults to fixtures directory with self_attn and mlp modules from layer 0"
        "of TheBloke/Llama-7B-GPTQ and TheBloke/Mistral-7B-v0.1-GPTQ"
        "Download first by running download_test_modules function in this file",
    )
    parser.add_argument("--model", type=str, default="llama", help="Saved model name")
    parser.add_argument(
        "--module",
        type=str,
        default="self_attn",
        help="Module name for benchmarking (must be saved in fixtures path)",
    )
    parser.add_argument(
        "--layer", type=str, default="q_proj", help="Layer name in module to benchmark"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check outputs against qlinear_cuda_old"
    )
    args = parser.parse_args()
    fixtures_path = args.fixtures_path
    if not os.path.exists(fixtures_path):
        os.makedirs(fixtures_path)
        download_test_modules(fixtures_path)

    qstate: GPTQState = get_qstate(fixtures_path, args.model, args.module, args.layer)
    run_bench(qstate, check=args.check)
