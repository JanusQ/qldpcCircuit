from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

# Get CUDA path
cuda_path = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if not os.path.exists(cuda_path):
    cuda_path = '/usr/local/cuda'  # fallback

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "enumlikelydecoder",
        [
            "pybind_wrapper.cpp",
            "qldpc_decoder.cpp",
            "qldpc_decoder_csc_simple.cpp"
        ],
        include_dirs=[
            pybind11.get_include()
        ],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[
            "-std=c++11",
            "-O3",
        ],
        extra_link_args=[],
        language="c++"
    ),
]

setup(
    name="enumlikelydecoder",
    version="1.0.0",
    author="QLDPC Team",
    author_email="",
    description="QLDPC Enumeration Decoder with CSC optimization support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
) 