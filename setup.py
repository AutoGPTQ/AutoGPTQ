import os
import sys
from pathlib import Path
from setuptools import setup, find_packages


common_setup_kwargs = {
    "version": "0.4.0",
    "name": "auto_gptq",
    "author": "PanQiWei",
    "description": "An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/PanQiWei/AutoGPTQ",
    "keywords": ["gptq", "quantization", "large-language-models", "transformers"],
    "platforms": ["windows", "linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12.0",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}


BUILD_CUDA_EXT = int(os.environ.get('BUILD_CUDA_EXT', '1')) == 1
if BUILD_CUDA_EXT:
    try:
        import torch
    except:
        print("Building cuda extension requires PyTorch(>=1.13.0) been installed, please install PyTorch first!")
        sys.exit(-1)

    CUDA_VERSION = None
    ROCM_VERSION = os.environ.get('ROCM_VERSION', None)
    if ROCM_VERSION and not torch.version.hip:
        print(
            f"Trying to compile auto-gptq for RoCm, but PyTorch {torch.__version__} "
            "is installed without RoCm support."
        )
        sys.exit(-1)

    if not ROCM_VERSION:
        default_cuda_version = torch.version.cuda
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if ROCM_VERSION:
        common_setup_kwargs['version'] += f"+rocm{ROCM_VERSION}"
    else:
        if not CUDA_VERSION:
            print(
                f"Trying to compile auto-gptq for CUDA, byt Pytorch {torch.__version__} "
                "is installed without CUDA support."
            )
            sys.exit(-1)
        common_setup_kwargs['version'] += f"+cu{CUDA_VERSION}"


requirements = [
    "accelerate>=0.19.0",
    "datasets",
    "numpy",
    "rouge",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.31.0",
    "peft"
]

extras_require = {
    "triton": ["triton==2.0.0"],
    "test": ["parameterized"]
}

include_dirs = ["autogptq_cuda"]

additional_setup_kwargs = dict()
if BUILD_CUDA_EXT:
    from torch.utils import cpp_extension

    if not ROCM_VERSION:
        from distutils.sysconfig import get_python_lib
        conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

        print("conda_cuda_include_dir", conda_cuda_include_dir)
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)
            print(f"appending conda cuda include dir {conda_cuda_include_dir}")
    extensions = [
        cpp_extension.CUDAExtension(
            "autogptq_cuda_64",
            [
                "autogptq_cuda/autogptq_cuda_64.cpp",
                "autogptq_cuda/autogptq_cuda_kernel_64.cu"
            ]
        ),
        cpp_extension.CUDAExtension(
            "autogptq_cuda_256",
            [
                "autogptq_cuda/autogptq_cuda_256.cpp",
                "autogptq_cuda/autogptq_cuda_kernel_256.cu"
            ]
        )
    ]

    if os.environ.get("INCLUDE_EXLLAMA_KERNELS", "1") == "1":  # TODO: improve github action to always compile exllama_kernels
        extensions.append(
            cpp_extension.CUDAExtension(
                "exllama_kernels",
                [
                    "autogptq_cuda/exllama/exllama_ext.cpp",
                    "autogptq_cuda/exllama/cuda_buffers.cu",
                    "autogptq_cuda/exllama/cuda_func/column_remap.cu",
                    "autogptq_cuda/exllama/cuda_func/q4_matmul.cu",
                    "autogptq_cuda/exllama/cuda_func/q4_matrix.cu"
                ]
            )
        )

    additional_setup_kwargs = {
        "ext_modules": extensions,
        "cmdclass": {'build_ext': cpp_extension.BuildExtension}
    }
common_setup_kwargs.update(additional_setup_kwargs)
setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_dirs=include_dirs,
    python_requires=">=3.8.0",
    **common_setup_kwargs
)
