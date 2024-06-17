<h1 align="center">AutoGPTQ</h1>
<p align="center">An easy-to-use LLM quantization package with user-friendly APIs, based on GPTQ algorithm (weight-only quantization).</p>
<p align="center">
    <a href="https://github.com/PanQiWei/AutoGPTQ/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/PanQiWei/AutoGPTQ.svg">
    </a>
    <a href="https://pypi.org/project/auto-gptq/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/auto-gptq">
    </a>
</p>
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/PanQiWei/AutoGPTQ/blob/main/README_zh.md">中文</a>
    </p>
</h4>

## News or Update

- 2024-06-XX - (News)   PENDING
- 2024-02-15 - (News) - AutoGPTQ 0.7.0 is released, with [Marlin](https://github.com/IST-DASLab/marlin) int4*fp16 matrix multiplication kernel support, with the argument `use_marlin=True` when loading models.
- 2023-08-23 - (News) - 🤗 Transformers, optimum and peft have integrated `auto-gptq`, so now running and training GPTQ models can be more available to everyone! See [this blog](https://huggingface.co/blog/gptq-integration) and it's resources for more details!

*For more histories please turn to [here](docs/NEWS_OR_UPDATE.md)*

## Performance Comparison

### Inference Speed
> The result is generated using [this script](examples/benchmark/generation_speed.py), batch size of input is 1, decode strategy is beam search and enforce the model to generate 512 tokens, speed metric is tokens/s (the larger, the better).
>
> The quantized model is loaded using the setup that can gain the fastest inference speed.

| model         | GPU           | num_beams | fp16  | gptq-int4 |
|---------------|---------------|-----------|-------|-----------|
| llama-7b      | 1xA100-40G    | 1         | 18.87 | 25.53     |
| llama-7b      | 1xA100-40G    | 4         | 68.79 | 91.30     |
| moss-moon 16b | 1xA100-40G    | 1         | 12.48 | 15.25     |
| moss-moon 16b | 1xA100-40G    | 4         | OOM   | 42.67     |
| moss-moon 16b | 2xA100-40G    | 1         | 06.83 | 06.78     |
| moss-moon 16b | 2xA100-40G    | 4         | 13.10 | 10.80     |
| gpt-j 6b      | 1xRTX3060-12G | 1         | OOM   | 29.55     |
| gpt-j 6b      | 1xRTX3060-12G | 4         | OOM   | 47.36     |


### Perplexity
For perplexity comparison, you can turn to [here](https://github.com/qwopqwop200/GPTQ-for-LLaMa#result) and [here](https://github.com/qwopqwop200/GPTQ-for-LLaMa#gptq-vs-bitsandbytes)

## Installation

AutoGPTQ is available on Linux only. You can install the latest stable release of AutoGPTQ from pip with pre-built wheels:

| CUDA/ROCm version | Installation                                                                                      | Built against PyTorch |
|-------------------|---------------------------------------------------------------------------------------------------|-----------------------|
| CUDA 12.1         | `pip install auto-gptq --no-build-isolation`                                                                            | 2.3.1+cu121           |


On NVIDIA systems, AutoGPTQ does not support [Maxwell or lower](https://qiita.com/uyuni/items/733a93b975b524f89f46) GPUs.

### Install from source

Clone the source code:
```bash
git clone https://github.com/Qubitium/AutoGPTQ.git && cd AutoGPTQ
```

A few packages are required in order to build from source: `pip install numpy gekko pandas`.

Then, install locally from source:
```bash
pip install -vvv --no-build-isolation -e .
```

As a last resort, if the above command fails, you can try `python setup.py install`.

## Quick Tour

### Quantization and Inference
> warning: this is just a showcase of the usage of basic apis in AutoGPTQ, which uses only one sample to quantize a much small model, quality of quantized model using such little samples may not good.

Below is an example for the simplest use of `auto_gptq` to quantize a model and inference after quantization:
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model
model.save_quantized(quantized_model_dir)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
```

For more advanced features of model quantization, please reference to [this script](examples/quantization/quant_with_alpaca.py)

### Customize Model
<details>

<summary>Below is an example to extend `auto_gptq` to support `OPT` model, as you will see, it's very easy:</summary>

```python
from auto_gptq.modeling import BaseGPTQForCausalLM


class OPTGPTQForCausalLM(BaseGPTQForCausalLM):
    # chained attribute name of transformer layer block
    layers_block_name = "model.decoder.layers"
    # chained attribute names of other nn modules that in the same level as the transformer layer block
    outside_layer_modules = [
        "model.decoder.embed_tokens", "model.decoder.embed_positions", "model.decoder.project_out",
        "model.decoder.project_in", "model.decoder.final_layer_norm"
    ]
    # chained attribute names of linear layers in transformer layer module
    # normally, there are four sub lists, for each one the modules in it can be seen as one operation,
    # and the order should be the order when they are truly executed, in this case (and usually in most cases),
    # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"]
    ]
```
After this, you can use `OPTGPTQForCausalLM.from_pretrained` and other methods as shown in Basic.

</details>

### Evaluation on Downstream Tasks
You can use tasks defined in `auto_gptq.eval_tasks` to evaluate model's performance on specific down-stream task before and after quantization.

The predefined tasks support all causal-language-models implemented in [🤗 transformers](https://github.com/huggingface/transformers) and in this project.

<details>

<summary>Below is an example to evaluate `EleutherAI/gpt-j-6b` on sequence-classification task using `cardiffnlp/tweet_sentiment_multilingual` dataset:</summary>

```python
from functools import partial

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.eval_tasks import SequenceClassificationTask


MODEL = "EleutherAI/gpt-j-6b"
DATASET = "cardiffnlp/tweet_sentiment_multilingual"
TEMPLATE = "Question:What's the sentiment of the given text? Choices are {labels}.\nText: {text}\nAnswer:"
ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
LABELS = list(ID2LABEL.values())


def ds_refactor_fn(samples):
    text_data = samples["text"]
    label_data = samples["label"]

    new_samples = {"prompt": [], "label": []}
    for text, label in zip(text_data, label_data):
        prompt = TEMPLATE.format(labels=LABELS, text=text)
        new_samples["prompt"].append(prompt)
        new_samples["label"].append(ID2LABEL[label])

    return new_samples


#  model = AutoModelForCausalLM.from_pretrained(MODEL).eval().half().to("cuda:0")
model = AutoGPTQForCausalLM.from_pretrained(MODEL, BaseQuantizeConfig())
tokenizer = AutoTokenizer.from_pretrained(MODEL)

task = SequenceClassificationTask(
        model=model,
        tokenizer=tokenizer,
        classes=LABELS,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="label",
        **{
            "num_samples": 1000,  # how many samples will be sampled to evaluation
            "sample_max_len": 1024,  # max tokens for each sample
            "block_max_len": 2048,  # max tokens for each data block
            # function to load dataset, one must only accept data_name_or_path as input
            # and return datasets.Dataset
            "load_fn": partial(datasets.load_dataset, name="english"),
            # function to preprocess dataset, which is used for datasets.Dataset.map,
            # must return Dict[str, list] with only two keys: [prompt_col_name, label_col_name]
            "preprocess_fn": ds_refactor_fn,
            # truncate label when sample's length exceed sample_max_len
            "truncate_prompt": False
        }
    )

# note that max_new_tokens will be automatically specified internally based on given classes
print(task.run())

# self-consistency
print(
    task.run(
        generation_config=GenerationConfig(
            num_beams=3,
            num_return_sequences=3,
            do_sample=True
        )
    )
)
```

</details>

## Learn More
[tutorials](docs/tutorial) provide step-by-step guidance to integrate `auto_gptq` with your own project and some best practice principles.

[examples](examples/README.md) provide plenty of example scripts to use `auto_gptq` in different ways.

## Supported Models

> you can use `model.config.model_type` to compare with the table below to check whether the model you use is supported by `auto_gptq`.
>
> for example, model_type of `WizardLM`, `vicuna` and `gpt4all` are all `llama`, hence they are all supported by `auto_gptq`.

| model type                         | quantization | inference | 
|------------------------------------|--------------|-----------|
| bloom                              | ✅            | ✅         |
| gpt2                               | ✅            | ✅         | 
| gpt_neox                           | ✅            | ✅         |
| gptj                               | ✅            | ✅         | 
| llama                              | ✅            | ✅         | 
| moss                               | ✅            | ✅         |
| opt                                | ✅            | ✅         |
| gpt_bigcode                        | ✅            | ✅         |
| codegen                            | ✅            | ✅         |
| falcon(RefinedWebModel/RefinedWeb) | ✅            | ✅         |

## Supported Evaluation Tasks
Currently, `auto_gptq` supports: `LanguageModelingTask`, `SequenceClassificationTask` and `TextSummarizationTask`; more Tasks will come soon!

## Running tests

Tests can be run with:

```
pytest tests/ -s
```

## FAQ

### Which kernel is used by default?

AutoGPTQ will use Marlin, followed by Exllama v2, Exallama V1, CUDA kernels in that order for maximum inference performance.

## Acknowledgement
- Special thanks **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq), and for releasing [Marlin kernel](https://github.com/IST-DASLab/marlin) for mixed precision computation.
- Special thanks **qwopqwop200**, for code in this project that relevant to quantization are mainly referenced from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).
- Special thanks to **turboderp**, for releasing [Exllama](https://github.com/turboderp/exllama) and [Exllama v2](https://github.com/turboderp/exllamav2) libraries with efficient mixed precision kernels.
