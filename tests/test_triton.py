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
MAX_SEQ_LEN = 10
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
        m = t.blocked_autorange()
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
    model_id=MODEL_ID,
    inject_fused_attention=False,
    inject_fused_mlp=False,
    **model_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        trainable=True,
        inject_fused_attention=inject_fused_attention,
        inject_fused_mlp=inject_fused_mlp,
        disable_exllamav2=True,
        disable_exllama=True,
        **model_kwargs,
    )

    model.warmup_triton()
    return model, tokenizer


def get_optimizer(model, lr=LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


def train_step(model, optimizer, data_loader, device="cuda", num_steps=10):
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


def setup(model_id, dataset_id, max_seq_length, batch_size):
    ref_model, tokenizer = get_model_and_tokenizer(
        model_id=model_id,
        use_triton=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    test_model, _ = get_model_and_tokenizer(
        model_id=model_id,
        use_tritonv2=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )

    data_loader = get_data_loader(
        dataset_id, tokenizer, max_length=max_seq_length, batch_size=batch_size
    )

    return ref_model, test_model, data_loader


# out = model(**batch)
def test_triton_qlinear():
    ref_model, _ = get_model_and_tokenizer(
        model_id=MODEL_ID,
        use_triton=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    test_model, _ = get_model_and_tokenizer(
        model_id=MODEL_ID,
        use_tritonv2=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    hidden_size = ref_model.model.model.embed_tokens.weight.shape[1]
    test_data = torch.randn((1, 2048, hidden_size), dtype=torch.float16).cuda()

    qlinear_ref = ref_model.model.model.layers[0].self_attn.q_proj
    qlinear_test = test_model.model.model.layers[0].self_attn.q_proj

    # test_batch = next(iter(data_loader))
    test_out = qlinear_test(test_data)
    ref_out = qlinear_ref(test_data)
    print(f"Mean diff: {torch.mean(torch.abs(test_out - ref_out))}")
    benchmark_forward(qlinear_ref, test_data, desc="Triton", verbose=True)
    benchmark_forward(qlinear_test, test_data, desc="Triton-v2", verbose=True)


test_triton_qlinear()
