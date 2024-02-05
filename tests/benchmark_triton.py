import argparse
import os

import torch
import torch.utils.benchmark as benchmark
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM

MODEL_ID = "TheBloke/Llama-7B-GPTQ"
DATASET_ID = "timdettmers/openassistant-guanaco"
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 2048
BATCH_SIZE = 5
NUM_TRAIN_STEPS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_diff(a, ref):
    eps = 1e-6
    return f"Maxdiff: {(a - ref).abs().max()}, Mean relative diff: {((a - ref).abs() / (ref.abs() + eps)).mean()}"


def benchmark_forward(
    fn,
    *inputs,
    repeats="auto",
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    if repeats == "auto":
        m = t.adaptive_autorange()
    else:
        m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats="auto",
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    if repeats == "auto":
        m = t.blocked_autorange()
    else:
        m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def get_hf_model(model_id=MODEL_ID, **model_kwargs):
    from transformers import AutoModelForCausalLM, GPTQConfig

    quantization_config = GPTQConfig(bits=4, use_exllama=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    return model


def get_model_and_tokenizer(
    model_id,
    use_triton=False,
    use_tritonv2=False,
    disable_exllama=True,
    disable_exllamav2=True,
    trainable=False,
    inject_fused_attention=False,
    inject_fused_mlp=False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        trainable=trainable,
        use_triton=use_triton,
        use_tritonv2=use_tritonv2,
        disable_exllamav2=disable_exllamav2,
        disable_exllama=disable_exllama,
        inject_fused_attention=inject_fused_attention,
        inject_fused_mlp=inject_fused_mlp,
    )
    if use_triton or use_tritonv2:
        model.warmup_triton()

    return model, tokenizer


def get_optimizer(model, lr=LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


def train_step(model, optimizer, data_loader, num_steps=10):
    # training and evaluation
    with torch.cuda.amp.autocast():
        total_loss = 0
        for batch in data_loader:
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(data_loader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{train_ppl=} {train_epoch_loss=}")


def get_data_loader(dataset_id, tokenizer, max_length=MAX_SEQ_LEN, batch_size=5):
    ds = load_dataset(dataset_id, split="train")

    def tokenize(element, dataset_text_field="text"):
        outputs = tokenizer(
            element[dataset_text_field],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def collate_fn(batch):
        batch = tokenizer.pad(batch, padding="longest", return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return {k: v.cuda() for k, v in batch.items()}

    tokenized_dataset = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=4,
    )

    data_loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return data_loader


def setup(model_id, use_triton=False, use_tritonv2=True):
    ref_model, tokenizer = get_model_and_tokenizer(
        model_id=model_id,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    assert not (use_triton and use_tritonv2), "Cannot use both triton and tritonv2"
    test_model, _ = get_model_and_tokenizer(
        model_id=model_id,
        use_triton=use_triton,
        use_tritonv2=use_tritonv2,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )

    # data_loader = get_data_loader(
    #     dataset_id, tokenizer, max_length=max_seq_length, batch_size=batch_size
    # )

    return ref_model, test_model  # , data_loader


# out = model(**batch)
def run_benchmark(
    model_id,
    kernel,
    max_seq_len,
    batch_size,
    use_triton=False,
    use_tritonv2=False,
    disable_exllama=True,
    disable_exllamav2=True,
):
    if kernel == "triton":
        print("Benchmarking triton kernel...")
        use_triton = True
    elif kernel == "tritonv2":
        print("Benchmarking tritonv2 kernel...")
        use_tritonv2 = True
    elif kernel == "exllama":
        print("Benchmarking exllama kernel...")
        disable_exllama = False
    elif kernel == "exllamav2":
        print("Benchmarking exllamav2 kernel...")
        disable_exllamav2 = False
    else:
        print("Benchmarking default kernel...")
    model, tokenizer = get_model_and_tokenizer(
        model_id=model_id,
        use_triton=use_triton,
        use_tritonv2=use_tritonv2,
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
    )
    data_loader = get_data_loader(
        dataset_id=DATASET_ID,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        batch_size=batch_size,
    )
    batch = next(iter(data_loader))
    benchmark_forward(
        model, **batch, desc="Tritonv2" if use_tritonv2 else "Triton", verbose=True
    )


def run_test(model_id, use_tritonv2=True, batch_size=1, max_seq_len=10, seed=3407):
    ref_model, test_model = setup(
        model_id=model_id,
        use_triton=not use_tritonv2,
        use_tritonv2=use_tritonv2,
    )
    torch.manual_seed(seed)
    hidden_size = ref_model.model.model.embed_tokens.weight.shape[1]
    test_data = torch.randn(
        (batch_size, max_seq_len, hidden_size), dtype=torch.float16
    ).cuda()

    for i, (ref_layer, test_layer) in enumerate(
        zip(ref_model.model.model.layers, test_model.model.model.layers)
    ):
        ref_attn, test_attn = ref_layer.self_attn, test_layer.self_attn

        for k in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            ref_out = getattr(ref_attn, k)(test_data)
            test_out = getattr(test_attn, k)(test_data)
            print(f"Layer {i} self_attn {k}: diff={get_diff(test_out, ref_out)}")
        print()
        ref_mlp, test_mlp = ref_layer.mlp, test_layer.mlp
        down_proj_input = torch.randn(
            batch_size, max_seq_len, ref_mlp.intermediate_size, dtype=torch.float16
        ).cuda()
        for k in ["gate_proj", "up_proj", "down_proj"]:
            if k == "down_proj":
                ref_out = getattr(ref_mlp, k)(down_proj_input)
                test_out = getattr(test_mlp, k)(down_proj_input)
            else:
                ref_out = getattr(ref_mlp, k)(test_data)
                test_out = getattr(test_mlp, k)(test_data)
                print(
                    f"Layer {i} mlp {k}: diff={get_diff(test_out, ref_out)}"
                )  # , get_diff(test_out.logits, ref_out.logits)
        print()
    # torch.testing.assert_allclose(test_out.logits, ref_out.logits, rtol=3e-5, atol=2e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Max sequence length for benchmarking",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for benchmarking"
    )
    parser.add_argument("--use_tritonv2", action="store_true", help="Use Tritonv2")
    parser.add_argument(
        "--benchmark_kernel",
        type=str,
        default="tritonv2",
        choices=["triton", "tritonv2", "exllama", "exllamav2", "default"],
        help="Run benchmark",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test triton vs triton-v2 outputs"
    )
    parser.add_argument(
        "--test_kernel",
        type=str,
        default="tritonv2",
        choices=["triton", "tritonv2"],
        help="Whether to compare triton or tritonv2 against cuda-old ref",
    )
    args = parser.parse_args()
    if args.test:
        use_tritonv2 = True if args.test_kernel == "tritonv2" else False
        use_triton = not use_tritonv2
        print(
            "Testing qlinear outputs between ref (cuda_old) vs {}".format(
                "Tritonv2" if use_tritonv2 else "Triton"
            )
        )

        run_test(
            args.model_id,
            use_tritonv2=use_tritonv2,
        )
    else:
        print("Running benchmark...")
        run_benchmark(
            args.model_id,
            kernel=args.benchmark_kernel,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
