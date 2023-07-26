import os
import platform
import sys
from pathlib import Path
from setuptools import setup, find_packages

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

python_min_version = (3, 8, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(f"You are using Python {platform.python_version()}. Python >={python_min_version_str} is required.")
    sys.exit(-1)

if TORCH_AVAILABLE:
    CUDA_VERSION = "".join(torch.version.cuda.split("."))
else:
    CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", "").split("."))

common_setup_kwargs = {
    "version": "0.3.2",
    "name": "auto_gptq",
    "author": "PanQiWei",
    "description": "An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/PanQiWei/AutoGPTQ",
    "keywords": ["gptq", "quantization", "large-language-models", "pytorch", "transformers"],
    "platforms": ["windows", "linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    "python_requires": f">={python_min_version_str}"
}

if CUDA_VERSION:
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
    "triton": ["triton>=2.0.0"]
}

include_dirs = ["autogptq_cuda"]

if TORCH_AVAILABLE:
    BUILD_CUDA_EXT = int(os.environ.get('BUILD_CUDA_EXT', '1')) == 1
    
    additional_setup_kwargs = dict()
    if BUILD_CUDA_EXT:
        from torch.utils import cpp_extension
        from distutils.sysconfig import get_python_lib
        conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
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
        **common_setup_kwargs
    )
else:
    setup(
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=include_dirs,
        **common_setup_kwargs
    )
