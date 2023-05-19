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
    <p>
</h4>

## News or Update
- 2023-05-12 - (In Progress) - `peft` + `auto-gptq` + multi-modal data = easily fine tune LLMs to gain multi-modal instruction following ability with low resources, stay tune!
- 2023-05-04 - (Update) - Support using faster cuda kernel when `not desc_act or group_size == -1`.
- 2023-04-29 - (Update) - Support loading quantized model from arbitrary quantize_config and model_basename.
- 2023-04-28 - (Update) - Support CPU offload and quantize/inference on multiple devices, support `gpt2` type models.

*For more histories please turn to [here](docs/NEWS_OR_UPDATE.md)*

## Installation

### Quick Installation
You can install the latest stable release of AutoGPTQ from pip:
```shell
pip install auto-gptq
```
#### disable cuda extensions
By default, cuda extensions will be installed when `torch` and `cuda` is already installed in your machine, if you don't want to use them, using:
```shell
BUILD_CUDA_EXT=0 pip install auto-gptq
```
And to make sure `quant_cuda` is not ever in your virtual environment, run:
```shell
pip uninstall quant_cuda -y
```
#### to support LLaMa model
For some people want to try LLaMa and whose `transformers` version not meet the newest one that supports it, using:
```shell
pip install auto-gptq[llama]
```
#### to support triton speedup
To integrate with `triton`, using:
> warning: currently triton only supports linux; 3-bit quantization is not supported when using triton

```shell
pip install auto-gptq[triton]
```

### Install from source
Clone the source code:
```shell
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
```
Then, install from source:
```shell
pip install .
```
Like quick installation, you can also set `BUILD_CUDA_EXT=0` to disable pytorch extension building.

Use `.[llama]` if you want to try LLaMa model.

Use `.[triton]` if you want to integrate with triton and it's available on your operating system.


## Supported Models
Currently, `auto_gptq` supports: `bloom`, `gpt2`, `gpt_neox`, `gptj`, `llama`, `moss` and `opt`; more Transformer models will come soon!

## Supported Evaluation Tasks
Currently, `auto_gptq` supports: `LanguageModelingTask`, `SequenceClassificationTask` and `TextSummarizationTask`; more Tasks will come soon!

## Usage

**Here are [tutorials](docs/tutorial)(continue updating...) for using `auto-gptq`, it's highly recommended for newcomers to read them first before trying example scripts.** 

### Basic
> warning: this is just a show case of the usage of basic apis in AutoGPTQ, which uses only one sample to quantize a much small model, thus may not performs as well as expected in LLMs.

Below is an example for the simplest use of `auto_gptq`: 
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
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples, use_triton=False)

# save quantized model
model.save_quantized(quantized_model_dir)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)


# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0"))[0]))

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
```

The following example demonstrates use of Hugging Face Hub for model downloading and uploading:
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=True # Use desc_act for higher inference quality from quantized model
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples, use_triton=False)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)

repo_id = f"YourUserName/{quantized_model_dir}"

# push quantized model to Hugging Face Hub. 
# To use use_auth_token=True, Login first via huggingface-cli login.
# Or pass explcit token with: use_auth_token="hf_xxxxxxx"
commit_message = f"AutoGPTQ model for {pretrained_model}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

# Alternatively you can save and push at the same time:
# model.push_to_hub(repo_id, quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)
# load quantized model to the first GPU

model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0"))[0]))
```

For more advanced features of model quantization, please reference to [this script](examples/quantization/quant_with_alpaca.py)

### Customize Model
Below is an example to extend `auto_gptq` to support `OPT` model, as you will see, it's very easy:
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

### Evaluation on Downstream Tasks
You can use tasks defined in `auto_gptq.eval_tasks` to evaluate model's performance on specific down-stream task before and after quantization.

The predefined tasks support all causal-language-models implemented in [🤗 transformers](https://github.com/huggingface/transformers) and in this project.

Below is an example to evaluate `EleutherAI/gpt-j-6b` on sequence-classification task using `cardiffnlp/tweet_sentiment_multilingual` dataset:
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

### More Examples
For more examples, please turn to [examples](examples/README.md)

## Acknowledgement
- Specially thanks **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq).
- Specially thanks **qwopqwop200**, for code in this project that relevant to quantization are mainly referenced from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).
