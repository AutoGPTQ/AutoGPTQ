# Examples

To run example scripts in this folder, one must first install `auto_gptq` as described in [this](../README.md)

## Quantization
> Commands in this chapter should be run under `quantization` folder.

### Basic Usage
To Execute `basic_usage.py`, using command like this:
```shell
python basic_usage.py
```

To Execute `basic_usage_with_wikitext2.py`, using command like this:
```shell
python basic_usage_with_wikitext2.py
```
> Note: There is about 0.6 ppl degrade on opt-125m model using AutoGPTQ, compared to GPTQ-for-LLaMa.

### Quantize with Alpaca
To Execute `quant_with_alpaca.py`, using command like this:
```shell
python quant_with_alpaca.py --pretrained_model_dir "facebook/opt-125m" --per_gpu_max_memory 4 --quant_batch_size 16
```

Use `--help` flag to see detailed descriptions for more command arguments.

The alpaca dataset used in here is a cleaned version provided by **gururise** in [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)

## Evaluation
> Commands in this chapter should be run under `evaluation` folder.

### Language Modeling Task
`run_language_modeling_task.py` script gives an example of using `LanguageModelingTask` to evaluate model's performance on language modeling task before and after quantization using `tatsu-lab/alpaca` dataset.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python run_language_modeling_task.py --base_model_dir PATH/TO/BASE/MODEL/DIR --quantized_model_dir PATH/TO/QUANTIZED/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.

### Sequence Classification Task
`run_sequence_classification_task.py` script gives an example of using `SequenceClassificationTask` to evaluate model's performance on sequence classification task before and after quantization using `cardiffnlp/tweet_sentiment_multilingual` dataset.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python run_sequence_classification_task.py --base_model_dir PATH/TO/BASE/MODEL/DIR --quantized_model_dir PATH/TO/QUANTIZED/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.

### Text Summarization Task
`run_text_summarization_task.py` script gives an example of using `TextSummarizationTask` to evaluate model's performance on text summarization task before and after quantization using `samsum` dataset.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python run_text_summarization_task.py --base_model_dir PATH/TO/BASE/MODEL/DIR --quantized_model_dir PATH/TO/QUANTIZED/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.

## Push To Hub
> Commands in this chapter should be run under `push_to_hub` folder.

You can upload and share your quantized model to Hugging Face Hub by using `push_to_hub` function.

`push_quantized_model_to_hf_hub.py` provide a simple example to upload quantized model, tokenizer and configs at once.

First, you need to login, run the following command in the virtual environment where Hugging Face Transformers is installed:
```shell
huggingface-cli login
```

Then run the script like this:
```shell
python push_quantized_model_to_hf_hub.py --quantized_model_dir PATH/TO/QUANTIZED/MODEL/DIR --tokenizer_dir PATH/TO/TOKENIZER/DIR --repo_id REPO/ID
```

Use `--help` flag to see detailed descriptions for more command arguments.

## Benchmark
> Commands in this chapter should be run under `benchmark` folder.

### Generation Speed
`generation_speed.py` script gives an example of how to benchmark the generations speed of pretrained and quantized models that `auto_gptq` supports, this benchmarks model generation speed in tokens/s metric.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python generation_speed.py --model_name_pr_path PATH/TO/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.

## PEFT
> Commands in this chapter should be run under `peft` folder.

### Lora
`peft_lora_clm.py` script gives an example of instruction tuning gptq quantized model's lora adapter using tools in `auto_gptq.utils.peft_utils` and `🤗 peft` on alpaca dataset.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python peft_lora_clm.py --model_name_or_path PATH/TO/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.

### AdaLora
`peft_adalora_clm.py` script gives an example of instruction tuning gptq quantized model's adalora adapter using tools in `auto_gptq.utils.peft_utils` and `🤗 peft` on alpaca dataset.

To execute this script, using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python peft_adalora_clm.py --model_name_or_path PATH/TO/MODEL/DIR
```

Use `--help` flag to see detailed descriptions for more command arguments.