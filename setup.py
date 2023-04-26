import os
from setuptools import setup, find_packages

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

version = "v0.1.0-dev"

requirements = [
    "datasets",
    "numpy",
    "rouge",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.26.1"
]

extras_require = {
    "llama": ["transformers>=4.28.0"],
    "triton": ["triton>=2.0.0"]
}


if TORCH_AVAILABLE:
    from torch.utils import cpp_extension

    BUILD_CUDA_EXT = int(os.environ.get('BUILD_CUDA_EXT', '1')) == 1

    extensions = [
        cpp_extension.CUDAExtension(
            "quant_cuda",
            [
                "quant_cuda/quant_cuda.cpp",
                "quant_cuda/quant_cuda_kernel.cu"
            ]
        )
    ]
    additional_setup_kwargs = dict()
    if BUILD_CUDA_EXT:
        additional_setup_kwargs = {
            "ext_modules": extensions,
            "cmdclass": {'build_ext': cpp_extension.BuildExtension}
        }
    setup(
        name="auto_gptq",
        packages=find_packages(),
        version=version,
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=["quant_cuda"],
        **additional_setup_kwargs
    )
else:
    setup(
        name="auto_gptq",
        packages=find_packages(),
        version=version,
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=["quant_cuda"]
    )
