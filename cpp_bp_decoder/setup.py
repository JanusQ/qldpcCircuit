from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "BPdecoder",
        ["bp_decoder.cpp", "python_bindings.cpp"],
        include_dirs=["."],
        cxx_std=11,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="BPdecoder",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pybind11>=2.6.0"
    ],
    description="Belief Propagation with Accumulated Likelihood Decoder for QLDPC Codes",
) 