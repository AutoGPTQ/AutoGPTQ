import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup


os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.8.0.dev0",
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
DISABLE_QIGEN = int(os.environ.get('DISABLE_QIGEN', '1')) == 1
COMPILE_MARLIN = int(os.environ.get('COMPILE_MARLIN', '1')) == 1
UNSUPPORTED_COMPUTE_CAPABILITIES = ['3.5', '3.7', '5.0', '5.2', '5.3']

def detect_local_sm_architectures():
    """
    Detect compute capabilities of one machine's GPUs as PyTorch does.

    Copied from https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/cpp_extension.py#L1962-L1976
    """
    arch_list = []

    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        supported_sm = [int(arch.split('_')[1])
                        for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
        # Capability of the device may be higher than what's supported by the user's
        # NVCC, causing compilation error. User's NVCC is expected to match the one
        # used to build pytorch, so we use the maximum supported capability of pytorch
        # to clamp the capability.
        capability = min(max_supported_sm, capability)
        arch = f'{capability[0]}.{capability[1]}'
        if arch not in arch_list:
            arch_list.append(arch)

    arch_list = sorted(arch_list)
    arch_list[-1] += '+PTX'
    return arch_list


if BUILD_CUDA_EXT:
    try:
        import torch
    except Exception as e:
        print(f"Building PyTorch CUDA extension requires PyTorch being installed, please install PyTorch first: {e}.\n NOTE: This issue may be raised due to pip build isolation system (ignoring local packages). Please use `--no-build-isolation` when installing with pip, and refer to https://github.com/AutoGPTQ/AutoGPTQ/pull/620 for more details.")
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

        torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if torch_cuda_arch_list is not None:
            torch_cuda_arch_list = torch_cuda_arch_list.replace(' ', ';')
            archs = torch_cuda_arch_list.split(';')

            requested_but_unsupported_archs = {arch for arch in archs if arch in UNSUPPORTED_COMPUTE_CAPABILITIES }
            if len(requested_but_unsupported_archs) > 0:
                raise ValueError(f"Trying to compile AutoGPTQ for CUDA compute capabilities {torch_cuda_arch_list}, but AutoGPTQ does not support the compute capabilities {requested_but_unsupported_archs} (AutoGPTQ requires Pascal or higher). Please fix your environment variable TORCH_CUDA_ARCH_LIST (Reference: https://github.com/pytorch/pytorch/blob/v2.2.2/setup.py#L135-L139).")
        else:
            local_arch_list = detect_local_sm_architectures()
            local_but_unsupported_archs = {arch for arch in local_arch_list if arch in UNSUPPORTED_COMPUTE_CAPABILITIES}
            if len(local_but_unsupported_archs) > 0:
                raise ValueError(f"PyTorch detected the compute capabilities {local_arch_list} for the NVIDIA GPUs on the current machine, but AutoGPTQ can not be built for compute capabilities {local_but_unsupported_archs} (AutoGPTQ requires Pascal or higher). Please set the environment variable TORCH_CUDA_ARCH_LIST (Reference: https://github.com/pytorch/pytorch/blob/v2.2.2/setup.py#L135-L139) with your necessary architectures.")

        # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
        if not PYPI_RELEASE:
            common_setup_kwargs['version'] += f"+cu{CUDA_VERSION}"

requirements = [
    "accelerate>=0.29.2",
    "datasets",
    "sentencepiece",
    "numpy",
    "rouge",
    "gekko",
    "torch>=1.13.0",
    "safetensors>=0.3.1",
    "transformers>=4.31.0",
    "peft>=0.5.0",
    "tqdm",
]

extras_require = {
    "triton": ["triton==2.0.0"],
    "test": ["pytest", "parameterized"],
    "quality": ["ruff==0.1.5"],
}

include_dirs = ["autogptq_cuda"]

additional_setup_kwargs = {}
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
        except subprocess.CalledProcessError:
            raise Exception("Generating QiGen kernels failed with the error shown above.")

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
            extensions.append(
                cpp_extension.CUDAExtension(
                    'autogptq_marlin_cuda',
                    [
                        'autogptq_extension/marlin/marlin_cuda.cpp',
                        'autogptq_extension/marlin/marlin_cuda_kernel.cu',
                        'autogptq_extension/marlin/marlin_repack.cu'
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
