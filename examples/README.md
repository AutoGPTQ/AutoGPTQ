# Examples

To run example scripts in this folder, one must first install `auto_gptq` as described in [this](../README.md)

## Basic Usage
Run the following code to execute `basic_usage.py`:
```shell
python basic_usage.py
```

## Quantize with Alpaca
To Run this script, one also need to install `datasets` via `pip install datasets`.

Then Execute `quant_with_alpaca.py` using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python quant_with_alpaca.py --pretrained_model_dir "facebook/opt-125m"
```

The alpaca dataset used in here is a cleaned version provided by **gururise** in [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)
