#!/usr/bin/env python3
"""
Setup script for the Mean-Field Decoder extension
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "meanfield_decoder",
        ["pybind_wrapper.cpp"],
        include_dirs=[np.get_include()],
        language='c++',
        cxx_std=17,  # Use C++17 for structured bindings
        extra_compile_args=[
            '-O3',           # Optimize for speed
            '-march=native', # Use native CPU instructions
            '-ffast-math',   # Fast math operations
            '-Wall',         # Enable warnings
            '-Wextra',       # Extra warnings
        ],
        extra_link_args=[
            '-O3',
        ],
    ),
]

# Package metadata
setup(
    name="meanfield_decoder",
    version="1.0.0",
    author="QLDPC Circuit Team",
    author_email="",
    description="Mean-Field Decoder for QLDPC Codes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="quantum error correction, LDPC codes, mean-field theory, decoding",
) 