import os
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
import subprocess
import math
import platform

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.7.0.dev0",
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
        "Environment :: GPU :: NVIDIA CUDA :: 12",
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


PYPI_RELEASE = os.environ.get('PYPI_RELEASE', None)
BUILD_CUDA_EXT = int(os.environ.get('BUILD_CUDA_EXT', '1')) == 1
DISABLE_QIGEN = int(os.environ.get('DISABLE_QIGEN', '0')) == 1
COMPILE_MARLIN = int(os.environ.get('COMPILE_MARLIN', '0')) == 1

if BUILD_CUDA_EXT:
    try:
        import torch
    except Exception as e:
        print(f"Building cuda extension requires PyTorch (>=1.13.0) being installed, please install PyTorch first: {e}")
        sys.exit(1)

    CUDA_VERSION = None
    ROCM_VERSION = os.environ.get('ROCM_VERSION', None)
    if ROCM_VERSION and not torch.version.hip:
        print(
            f"Trying to compile auto-gptq for ROCm, but PyTorch {torch.__version__} "
            "is installed without ROCm support."
        )
        sys.exit(1)

    if not ROCM_VERSION:
        default_cuda_version = torch.version.cuda
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if ROCM_VERSION:
        common_setup_kwargs['version'] += f"+rocm{ROCM_VERSION}"
    else:
        if not CUDA_VERSION:
            print(
                f"Trying to compile auto-gptq for CUDA, but Pytorch {torch.__version__} "
                "is installed without CUDA support."
            )
            sys.exit(1)

        # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
        if not PYPI_RELEASE:
            common_setup_kwargs['version'] += f"+cu{CUDA_VERSION}"

requirements = [
    "accelerate>=0.26.0",
    "datasets",
    "sentencepiece",
    "numpy",
    "rouge",
    "gekko",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.31.0",
    "peft>=0.5.0",
    "tqdm",
]

extras_require = {
    "triton": ["triton==2.0.0"],
    "test": ["pytest", "parameterized"]
}

include_dirs = ["autogptq_cuda"]

additional_setup_kwargs = dict()
if BUILD_CUDA_EXT:
    from torch.utils import cpp_extension
       
    if platform.system() != "Windows" and platform.machine() != "aarch64" and not DISABLE_QIGEN:
        print("Generating qigen kernels...")
        cores_info = subprocess.run("cat /proc/cpuinfo | grep cores | head -1", shell=True, check=True, text=True, stdout=subprocess.PIPE).stdout.split(" ")
        if (len(cores_info) == 3 and cores_info[1].startswith("cores")) or (len(cores_info) == 2):
            p = int(cores_info[-1])
        else:
            p = os.cpu_count()
        try:
            subprocess.check_output(["python", "./autogptq_extension/qigen/generate.py", "--module", "--search", "--p", str(p)])
        except subprocess.CalledProcessError as e:
            raise Exception(f"Generating QiGen kernels failed with the error shown above.")

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
                "autogptq_extension/cuda_64/autogptq_cuda_64.cpp",
                "autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu"
            ]
        ),
        cpp_extension.CUDAExtension(
            "autogptq_cuda_256",
            [
                "autogptq_extension/cuda_256/autogptq_cuda_256.cpp",
                "autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu"
            ]
        )
    ]
    
    if platform.system() != "Windows":
        if platform.machine() != "aarch64" and not DISABLE_QIGEN:
            extensions.append(
                cpp_extension.CppExtension(
                    "cQIGen",
                    [
                        'autogptq_extension/qigen/backend.cpp'
                    ],
                    extra_compile_args = ["-O3", "-mavx", "-mavx2", "-mfma", "-march=native", "-ffast-math", "-ftree-vectorize", "-faligned-new", "-std=c++17", "-fopenmp", "-fno-signaling-nans", "-fno-trapping-math"]
                )
            )

        # Marlin is not ROCm-compatible, CUDA only
        if not ROCM_VERSION and COMPILE_MARLIN:
            torch_cuda_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", None)

            if not torch_cuda_archs:
                raise ValueError('The environment variable `TORCH_CUDA_ARCH_LIST` needs to be specified to compile AutoGPTQ with Marlin kernel. Example: `TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"`.')

            archs_list = torch_cuda_archs.split(" ")
            if any(arch.startswith("6") or arch.startswith("7") for arch in archs_list):
                raise ValueError('Marlin kernel can not be compiled CUDA compute capability <8.0. Please specifiy a correct `TORCH_CUDA_ARCH_LIST` environment variable. Example: `TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"`')

            extensions.append(
                cpp_extension.CUDAExtension(
                    'marlin_cuda',
                    [
                        'autogptq_extension/marlin/marlin_cuda.cpp',
                        'autogptq_extension/marlin/marlin_cuda_kernel.cu'
                    ]
                )
            )

    if os.name == "nt":
        # On Windows, fix an error LNK2001: unresolved external symbol cublasHgemm bug in the compilation
        cuda_path = os.environ.get("CUDA_PATH", None)
        if cuda_path is None:
            raise ValueError("The environment variable CUDA_PATH must be set to the path to the CUDA install when installing from source on Windows systems.")
        extra_link_args = ["-L", f"{cuda_path}/lib/x64/cublas.lib"]
    else:
        extra_link_args = []

    extensions.append(
        cpp_extension.CUDAExtension(
            "exllama_kernels",
            [
                "autogptq_extension/exllama/exllama_ext.cpp",
                "autogptq_extension/exllama/cuda_buffers.cu",
                "autogptq_extension/exllama/cuda_func/column_remap.cu",
                "autogptq_extension/exllama/cuda_func/q4_matmul.cu",
                "autogptq_extension/exllama/cuda_func/q4_matrix.cu"
            ],
            extra_link_args=extra_link_args
        )
    )
    extensions.append(
        cpp_extension.CUDAExtension(
            "exllamav2_kernels",
            [
                "autogptq_extension/exllamav2/ext.cpp",
                "autogptq_extension/exllamav2/cuda/q_matrix.cu",
                "autogptq_extension/exllamav2/cuda/q_gemm.cu",
            ],
            extra_link_args=extra_link_args
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
