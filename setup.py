from os.path import abspath, dirname, join
from setuptools import setup, find_packages, Extension

from torch.utils import cpp_extension

project_root = dirname(abspath(__file__))

requirements = [
    "numpy",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.26.1"
]

extras_require = {
    "llama": ["transformers>=4.28.0"]
}

extensions = [
    cpp_extension.CUDAExtension(
        "quant_cuda",
        [
            join(project_root, "auto_gptq/quantization/quant_cuda.cpp"),
            join(project_root, "auto_gptq/quantization/quant_cuda_kernel.cu")
        ]
    )
]

setup(
    name="auto_gptq",
    packages=find_packages(),
    version="v0.0.1-dev",
    install_requires=requirements,
    extras_require=extras_require,
    ext_modules=extensions,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
