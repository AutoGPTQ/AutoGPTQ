# Advanced Model Loading and Best Practice
Welcome to the tutorial of AutoGPTQ, in this chapter, you will learn advanced model loading and best practice in `auto-gptq`.

## Arguments Introduction
In previous chapter, you learned how to load model into CPU or single GPU with the two basic apis:
- `.from_pretrained`: by default, load the whole pretrained model into CPU.
- `.from_quantized`: by default, load the whole quantized model into CPU, one can set `device='cuda'` to load model into a single GPU.

However, the default settings above may not meet many users' demands, for they want to try really large models but haven't enough CPU/GPU memory.

Luckily, in AutoGPTQ, we provide two advanced arguments that users can tweak based on the memory of hardware:
- `max_memory`: an optional `List[Dict[Union[str, int], str]]` type argument, can be used both in `.from_pretrained` and `.from_quantized`.
- `device_map`: an optional `str` type argument, currently only be supported in `.from_quantized`.

Before `auto-gptq`'s existence, there are many users have already used other popular tools such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) to quantize their model and saved with different name without `quantize_config.json` file introduced in previous chapter.

To address this, two more arguments were introduced in `.from_quantized` so that users can load quantized model with arbitrary names.
- `quantize_config`: an optional `BaseQuantizeConfig` type argument, can be used to match model file and initialize model incase `quantize_config.json` not in the directory where model is saved.
- `model_basename`: an optional `str` type argument, if specified, will be used to match model instead of using the file name format introduced in previous chapter.

## Multiple Devices Model Loading

### max_memory
With this argument, you can specify how much memory for CPU and GPUs to use at most.

That means, by specify the maximum CPU memory used at model loading, you can load some model weights to CPU and picked into GPU only when they're required to be used, and back CPU again after that. This is called "CPU offload", a very useful strategy that used when there is no room left for quantization or inference if you keep the whole model in GPU(s).

Assume you have multiple GPUs, for each of them, you can also specify maximum memory that used to load model, separately. And by this, quantization and inference will be executed across devices.

To better understanding, below are some examples.

```python
max_memory = {0: "20GIB"}
```
In this case, only first GPU (even if you have more GPUs) will be used to load model, and an error will be raised if the model requires memory over 20GB.

```python
max_memory = {0: "20GIB", 1: "20GIB"}
```
In this case, you can load model that smaller than 40GB into two GPUs, and the model will be split evenly.

```python
max_memory = {0: "10GIB", 1: "30GIB"}
```
In this case, you can also load model that smaller than 40GB into two GPUs, but the first GPU will use 10GB at most, which means if the model larger than 20GB, all model weights except the first 10GB will be loaded into the second GPU.

```python
max_memory = {0: "20GIB", "cpu": "20GIB"}
```
In this case, you can also load model that smaller than 40GB but the rest 20GB will be kept in CPU memory, only be collected into GPU when needed.

### device_map
So far, only `.from_quantized` supports this argument. You can specify it to use pre-set model loading strategies. Because under the hood, modules in model will be mapped to different devices based on the given `max_memory`, it's more convenient to use `device_map` directly if you don't want to spend much time on calculating how much memory in each device should be use to load model.

In the simplest way, you can set `device_map='auto'` and let 🤗 Accelerate handle the device map computation. For more pre-set strategies, you can reference to [this document](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).

## Best Practice

### At Quantization
It's always recommended to first consider loading the whole model into GPU(s) for it can save the time spend on transferring module's weights between CPU and GPU.

However, not everyone have large GPU memory. Roughly speaking, always specify the maximum memory CPU will be used to load model, then, for each GPU, you can preserve memory that can fit in 1~2(2~3 for the first GPU incase CPU offload used) model layers for examples' tensors and calculations in quantization, and load model weights using all others left. By this, all you need to do is a simple math based on the number of GPUs you have, the size of model weights file(s) and the number of model layers.

### At Inference
For inference, following this principle: always using single GPU if you can, otherwise multiple GPUs, CPU offload is the last one to consider.

## Conclusion
Congrats! You learned the advanced strategies to load model using `.from_pretrained` and `.from_quantized` in `auto-gptq` with some best practice advices. In the next chapter, you will learn how to quickly customize an AutoGPTQ model and use it to quantize and inference.