# AutoGPTQ
Automatically compress almost all Causal LMs in transformers using GPTQ algorithm.

## Installation
### Install from source
First, clone the source code:
```shell
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
```
Then, install from source:
```shell
pip instal .
```
For some people want to try LLaMa and whose `transformers` version not meet the newest one that supports it, using:
```shell
pip install .[llama]
```
## Acknowledgement
- Specially thanks **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq).
- Specially thanks **qwopqwop200**, for code in this project that relevant to quantization are mainly referenced from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda). 