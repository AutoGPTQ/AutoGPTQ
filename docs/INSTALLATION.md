# Installation

On Linux and Windows, AutoGPTQ can be installed through pre-built wheels for specific PyTorch versions:

| AutoGPTQ version | CUDA/ROCm version | Installation                                                                                               | Built against PyTorch |
|------------------|-------------------|------------------------------------------------------------------------------------------------------------|-----------------------|
| latest (0.7.1)   | CUDA 11.8         | `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`          | 2.2.1+cu118           |
| latest (0.7.1)   | CUDA 12.1         | `pip install auto-gptq`                                                                                    | 2.2.1+cu121           |
| latest (0.7.1)   | ROCm 5.7          | `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm571/`        | 2.2.1+rocm5.7         |
| 0.7.0   | CUDA 11.8         | `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`          | 2.2.0+cu118           |
| 0.7.0   | CUDA 12.1         | `pip install auto-gptq`                                                                                    | 2.2.0+cu121           |
| 0.7.0   | ROCm 5.7          | `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm571/`        | 2.2.0+rocm5.7         |
| 0.6.0            | CUDA 11.8         | `pip install auto-gptq==0.6.0 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`   | 2.1.1+cu118           |
| 0.6.0            | CUDA 12.1         | `pip install auto-gptq==0.6.0`                                                                             | 2.1.1+cu121           |
| 0.6.0            | ROCm 5.6          | `pip install auto-gptq==0.6.0 --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm561/` | 2.1.1+rocm5.6         |
| 0.5.1            | CUDA 11.8         | `pip install auto-gptq==0.5.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`   | 2.1.0+cu118           |
| 0.5.1            | CUDA 12.1         | `pip install auto-gptq==0.5.1`                                                                             | 2.1.0+cu121           |
| 0.5.1            | ROCm 5.6          | `pip install auto-gptq==0.5.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm561/` | 2.1.0+rocm5.6         |

AutoGPTQ is not available on macOS.