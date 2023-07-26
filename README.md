<h1 align="center">AutoGPTQ</h1>
<p align="center">An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.</p>
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

*<center>📣 Long time no see! 👋 Architecture upgrade, performance optimization and more new features will come in July and August, stay tune! 🥂</center>*

## News or Update

- 2023-07-26 - (Update) - An elegant [PPL benchmark script](examples/benchmark/perplexity.py) to get results that can be fairly compared with other libraries such as `llama.cpp`.
- 2023-06-05 - (Update) - Integrate with 🤗 peft to use gptq quantized model to train adapters, support LoRA, AdaLoRA, AdaptionPrompt, etc.
- 2023-05-30 - (Update) - Support download/upload quantized model from/to 🤗 Hub.
- 2023-05-27 - (Update) - Support quantization and inference for `gpt_bigcode`, `codegen` and `RefineWeb/RefineWebModel`(falcon) model types.
- 2023-05-04 - (Update) - Support using faster cuda kernel when `not desc_act or group_size == -1`.

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

### Quick Installation
You can install the latest stable release of AutoGPTQ from pip:
```shell
pip install auto-gptq
```
Start from v0.2.0, you can download pre-build wheel that satisfied your environment setup from each version's release assets and install it to skip building stage for the fastest installation speed. For example:
```shell
# firstly, cd the directory where the wheel saved, then execute command below
pip install auto_gptq-0.2.0+cu118-cp310-cp310-linux_x86_64.whl # install v0.2.0 auto_gptq pre-build wheel for linux in an environment whose python=3.10 and cuda=11.8
```
#### disable cuda extensions
By default, cuda extensions will be installed when `torch` and `cuda` is already installed in your machine, if you don't want to use them, using:
```shell
BUILD_CUDA_EXT=0 pip install auto-gptq
```
And to make sure `autogptq_cuda` is not ever in your virtual environment, run:
```shell
pip uninstall autogptq_cuda -y
```

#### to support triton speedup
To integrate with `triton`, using:
> warning: currently triton only supports linux; 3-bit quantization is not supported when using triton

```shell
pip install auto-gptq[triton]
```

### Install from source
<details>
<summary>click to see details</summary>

Clone the source code:
```shell
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
```
Then, install from source:
```shell
pip install .
```
Like quick installation, you can also set `BUILD_CUDA_EXT=0` to disable pytorch extension building.

Use `.[triton]` if you want to integrate with triton and it's available on your operating system.

</details>

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

# push quantized model to Hugging Face Hub. 
# to use use_auth_token=True, Login first via huggingface-cli login.
# or pass explcit token with: use_auth_token="hf_xxxxxxx"
# (uncomment the following three lines to enable this feature)
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

# alternatively you can save and push at the same time
# (uncomment the following three lines to enable this feature)
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

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

| model type                         | quantization | inference | peft-lora | peft-ada-lora | peft-adaption_prompt                                                                            |
|------------------------------------|--------------|-----------|-----------|---------------|-------------------------------------------------------------------------------------------------|
| bloom                              | ✅            | ✅         | ✅         | ✅             |                                                                                                 |
| gpt2                               | ✅            | ✅         | ✅         | ✅             |                                                                                                 |
| gpt_neox                           | ✅            | ✅         | ✅         | ✅             | ✅[requires this peft branch](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| gptj                               | ✅            | ✅         | ✅         | ✅             | ✅[requires this peft branch](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| llama                              | ✅            | ✅         | ✅         | ✅             | ✅                                                                                               |
| moss                               | ✅            | ✅         | ✅         | ✅             | ✅[requires this peft branch](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| opt                                | ✅            | ✅         | ✅         | ✅             |                                                                                                 |
| gpt_bigcode                        | ✅            | ✅         | ✅         | ✅             |                                                                                                 |
| codegen                            | ✅            | ✅         | ✅         | ✅             |                                                                                                 |
| falcon(RefinedWebModel/RefinedWeb) | ✅            | ✅         | ✅         | ✅             |                                                                                                 |

## Supported Evaluation Tasks
Currently, `auto_gptq` supports: `LanguageModelingTask`, `SequenceClassificationTask` and `TextSummarizationTask`; more Tasks will come soon!

## Acknowledgement
- Specially thanks **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq).
- Specially thanks **qwopqwop200**, for code in this project that relevant to quantization are mainly referenced from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).


[![Star History Chart](https://api.star-history.com/svg?repos=PanQiwei/AutoGPTQ&type=Date)](https://star-history.com/#PanQiWei/AutoGPTQ&Date)