from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "MarginalBPDecoder",
        ["python_bindings.cpp", "marginal_bp_decoder.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
        language="c++"
    ),
]

setup(
    name="marginal_bp_decoder",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
) 