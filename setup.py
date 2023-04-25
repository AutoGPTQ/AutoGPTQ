from setuptools import setup, find_packages

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

    extensions = [
        cpp_extension.CUDAExtension(
            "quant_cuda",
            [
                "quant_cuda/quant_cuda.cpp",
                "quant_cuda/quant_cuda_kernel.cu"
            ]
        )
    ]
    setup(
        name="auto_gptq",
        packages=find_packages(),
        version="v0.0.4-dev",
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=["quant_cuda"],
        ext_modules=extensions,
        cmdclass={'build_ext': cpp_extension.BuildExtension}
    )
else:
    setup(
        name="auto_gptq",
        packages=find_packages(),
        version="v0.0.4",
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=["quant_cuda"]
    )
