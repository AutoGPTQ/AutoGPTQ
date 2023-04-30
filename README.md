# AutoGPTQ
An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.

## News or Update
- 2023-04-29 - (Update) - Support loading quantized model from arbitrary quantize_config and model_basename.
- 2023-04-28 - (Update) - Support CPU offload and quantize/inference on multiple devices, support `gpt2` type models.
- 2023-04-26 - (Update) - Using `triton` to speed up inference is now supported.
- 2023-04-25 - (News&Update) - [MOSS](https://github.com/OpenLMLab/MOSS) is an open-source tool-augmented conversational language model from Fudan University, quantization is now supported in AutoGPTQ.
- 2023-04-23 - (Update) - Support evaluation on multiple (down-stream) tasks such as: language-modeling, text-classification, text-summarization.
- 2023-04-22 - (News) - qwopqwop200's [AutoGPTQ-triton](https://github.com/qwopqwop200/AutoGPTQ-triton) provides faster speed to integrate with quantized model, for everyone who can access to triton, try and enjoy yourself!
- 2023-04-20 - (News) - AutoGPTQ is automatically compatible with Stability-AI's newly released `gpt_neox` type model family [StableLM](https://github.com/Stability-AI/StableLM).
- 2023-04-16 - (Update) - Support quantization and inference for `bloom`, `gpt_neox`, `gptj`, `llama` and `opt`.

## Installation

### Quick Installation
You can install the latest stable release of AutoGPTQ from pip:
```shell
pip install auto-gptq
```
By default, pytorch extensions will be installed when `torch` is already in your virtual environment, if you don't want to use cuda extensions, using:
```shell
BUILD_CUDA_EXT=0 pip install auto-gptq
```
For some people want to try LLaMa and whose `transformers` version not meet the newest one that supports it, using:
```shell
pip install auto-gptq[llama]
```
To integrate with `triton`, using:
> warning: 3-bit quantization is not supported when using triton

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

### Basic
> warning: this is just a show case of the usage of basic apis in AutoGPTQ, which uses only one sample to quantize a much small model, thus may not performs as well as expected in LLMs.

Below is an example for the simplest use of auto_gptq: 
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


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

# load un-quantized model, the model will always be force loaded into cpu
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
# with value under torch.LongTensor type.
model.quantize(examples, use_triton=False)

# save quantized model
model.save_quantized(quantized_model_dir)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)

# load quantized model, currently only support cpu or single gpu
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to("cuda:0"))[0]))

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
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
After this, you can use `OPTGPTQForCausalLM.from_pretrained` and other functions

### Evaluation on Downstream Tasks
One can use tasks defined in `auto_gptq.eval_tasks` to evaluate model's performance on specific down-stream task before and after quantization.

The predefined tasks support all causal-language-models implemented in [Hugging Face transformers](https://github.com/huggingface/transformers) and in this project.

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

## Side Notes
### VRAM
Currently, I put everything (data, model, etc.) into CPU util one is required to be used or executed on GPU (and will back to CPU once the execution finished). Though I didn't run any benchmark to this date, but the maximum VRAM usage for GPTJ is about 6GB, which may be considered as a reference.

## Acknowledgement
- Specially thanks **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq).
- Specially thanks **qwopqwop200**, for code in this project that relevant to quantization are mainly referenced from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).
