CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall
PYTHON = python3
PYBIND11_INCLUDE = $(shell python3 -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Default target
all: MarginalBPDecoder.cpython-*.so

# Build using setuptools (recommended)
MarginalBPDecoder.cpython-*.so: python_bindings.cpp marginal_bp_decoder.cpp marginal_bp_decoder.hpp
	$(PYTHON) setup.py build_ext --inplace

# Clean build artifacts
clean:
	rm -rf build/
	rm -f MarginalBPDecoder.cpython-*.so
	rm -rf *.egg-info/

# Install dependencies
install-deps:
	pip install pybind11 numpy

# Test the build
test: MarginalBPDecoder.cpython-*.so
	$(PYTHON) -c "import MarginalBPDecoder; print('Build successful!')"

.PHONY: all clean install-deps test 